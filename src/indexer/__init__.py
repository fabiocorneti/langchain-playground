"""
Indexing functions.
"""
import logging
import os
from glob import glob

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WikipediaLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from configuration import EmbeddingsGenerationMode
from configuration import Splitter
from splitter import SpacyLimitTextSplitter
import settings
from store import ElasticsearchDataStore


def index() -> None:
    """
    Start indexing process
    """
    store = ElasticsearchDataStore()
    index_documents(store)


def clear() -> None:
    """
    Nukes data.
    """
    store = ElasticsearchDataStore()
    store.clear()


def index_documents(store: ElasticsearchDataStore) -> None:
    """
    Indexes documents from the docs directory.
    """
    chunk_kwargs = {
        "chunk_size": settings.CONFIGURATION.indexer.chunkSize,
        "chunk_overlap": settings.CONFIGURATION.indexer.chunkOverlap
    }
    embeddings_generation_mode = settings.CONFIGURATION.embeddingsGenerationMode
    indexer_splitter = settings.CONFIGURATION.indexer.splitter

    if indexer_splitter == Splitter.SPACY:
        splitter = SpacyLimitTextSplitter(token_limit=settings.CONFIGURATION.indexer.maxTokens,
                                          **chunk_kwargs)
    elif embeddings_generation_mode == EmbeddingsGenerationMode.OPENAI:
        chunk_kwargs["encoding_name"] = "cl100k_base"
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(**chunk_kwargs)
    else:
        splitter = RecursiveCharacterTextSplitter(**chunk_kwargs)

    for doc in glob("*", root_dir=settings.DOCS_DIR):
        doc_path = str(settings.DOCS_DIR / doc)
        result = os.path.splitext(doc_path)
        if result and len(result) > 1:
            extension = result[1]
            loader = None
            if extension == ".pdf":
                loader = PyPDFLoader(doc_path)
            elif extension == ".txt":
                loader = TextLoader(doc_path, encoding="utf-8")
            elif extension == ".wikipedia":
                with open(doc_path, encoding="utf-8") as query_file:
                    lines = query_file.readlines()
                    loader = WikipediaLoader(query=lines[0].strip(),
                                             load_max_docs=30,
                                             doc_content_chars_max=80000)
            if loader is None:
                logging.info("No loader configured for extension %s", extension)
            else:
                logging.info("Splitting document %s", doc_path)
                chunks = loader.load_and_split(text_splitter=splitter)
                logging.info("Splitted document %s", doc_path)
                store.index_chunks(doc, chunks)
        else:
            logging.info("Skipping document %s", doc_path)


__all__ = ['index']
