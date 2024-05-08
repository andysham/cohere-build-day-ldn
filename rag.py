import os
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
import re
from pick import pick
import subprocess

from typing import TypeVar

import cohere
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

pdfs = list((Path(__file__).parent / "pdfs").glob("*.pdf"))
co = cohere.Client(COHERE_API_KEY)
embeddings_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-multilingual-v3.0")

print("Loading pdfs into vector db...")

def remove_excess_newlines(s: str) -> str:
    return re.sub(r'\n(?:\w*\n)*', ' ', s)

def concat_documents(docs: list[Document]) -> Document:
    page_contents = "\n".join(doc.page_content for doc in docs)
    # metadata = docs[0].metadata
    return Document(page_content=page_contents)

def documents_from_pdf(pdf: Path) -> list[Document]:
    # print(pdf)
    loader = PyPDFLoader(pdf)
    return loader.load_and_split()

def document_from_pdf(pdf: Path) -> Document:
    return concat_documents(documents_from_pdf(pdf))

all_documents = list(document_from_pdf(pdf) for pdf in pdfs)
db = Chroma.from_documents(all_documents, embeddings_model)

@dataclass
class DocumentWithRanking:
    document: Document
    relevance_score: float

    @staticmethod
    def aggregate(*dwrs: "DocumentWithRanking") -> "DocumentWithRanking":
        page_content = dwrs[0].document.page_content
        for dwr in dwrs:
            assert page_content == dwr.document.page_content
        return DocumentWithRanking(
            document=dwrs[0].document,
            relevance_score=sum(dwr.relevance_score for dwr in dwrs) / len(dwrs) # average scores
        )


def get_candidate_documents(docs: list[Document], top_n: int) -> list[DocumentWithRanking]:

    documents_with_rankings: list[DocumentWithRanking] = []

    for good_doc in good_documents:

        docs: list[Document] = db.similarity_search(
            good_doc.page_content, 
            k=top_n * 10
        )

        docstrings = [remove_excess_newlines(doc.page_content) for doc in docs]
        # docstrings = [doc.page_content for doc in docs]

        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=good_doc.page_content,
            documents=docstrings,
            top_n=top_n
        )

        for results_item in response.results:
            document = docs[results_item.index]
            documents_with_rankings.append(
                DocumentWithRanking(
                    document=document,
                    relevance_score=results_item.relevance_score
                )
            )

    aggregated_documents_with_ranking: list[DocumentWithRanking] = []

    for _, _documents_with_rankings in groupby(
        sorted(
            documents_with_rankings, 
            key=lambda dwr: dwr.document.page_content, 
        ),
        key=lambda dwr: dwr.document.page_content
    ):

        aggregated_documents_with_ranking.append(
            DocumentWithRanking.aggregate(*_documents_with_rankings)
        )

    aggregated_documents_with_ranking = sorted(
        aggregated_documents_with_ranking, 
        key=lambda dwr: dwr.relevance_score, 
    )

    return aggregated_documents_with_ranking[:top_n]
    


def pdf_from_document(doc: Document) -> Path:
    return pdfs[[
        idx 
        for idx, d in enumerate(all_documents) 
        if d.page_content == doc.page_content
    ][0]]

def select_document(
    _documents: list[Document]
) -> Document:
    _pdfs = [pdf_from_document(d) for d in _documents]
    while True:
        _, index = pick(list(pdf.stem for pdf in _pdfs), "Please select a good pdf")
        pdf = _pdfs[index]
        subprocess.run(["code", str(pdf)])
        good_pdf_yn = input("Is this a good pdf (y/n)?")
        if good_pdf_yn.lower() == "y":
            break
    return _documents[index]


good_documents: list[Document] = [select_document(all_documents)]

while True:
    candidates = get_candidate_documents(good_documents, top_n=5)
    # candidate_pdfs: list[Path] = [pdf_from_document(c.document) for c in candidates]
    # _, index = pick(list(pdf.stem for pdf in candidate_pdfs), "Please select a similar document")
    good_documents.append(
        select_document([c.document for c in candidates])
    )


