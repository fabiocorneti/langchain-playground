"""
Elasticsearch store.
"""
import logging
from typing import List
import re

from elasticsearch import Elasticsearch
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.vectorstores import ElasticsearchStore
from langchain.vectorstores.utils import DistanceStrategy

from configuration import EmbeddingsGenerationMode
import settings

# Number of documents to return
K = 20
# Number of candidate documents to fetch per shard from the HNSW index
FETCH_K = 200

PROMPT_TEMPLATE = """
Use the following context to answer the question at the end.
If you don't know the answer, just say that you don't know,
don't try to make it up.

{context}

Question: {question}

Answer in Markdown format.
"""
PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


class ElasticsearchDataStore:
    """
    Handles Elasticsearch operations.
    """

    __client: Elasticsearch
    __store: ElasticsearchStore
    __qa: RetrievalQA

    def __init__(self) -> None:
        self.__client = Elasticsearch(
            hosts=[settings.CONFIGURATION.elasticsearch.url],
            basic_auth=(settings.CONFIGURATION.elasticsearch.username,
                        settings.CONFIGURATION.elasticsearch.password),
            request_timeout=settings.CONFIGURATION.elasticsearch.requestTimeout,
        )

        embeddings_mode = settings.CONFIGURATION.embeddingsGenerationMode
        normalize = settings.CONFIGURATION.elasticsearch.similarity == \
            DistanceStrategy.DOT_PRODUCT
        retrieval_strategy = ElasticsearchStore.ApproxRetrievalStrategy(hybrid=False)
        if embeddings_mode == EmbeddingsGenerationMode.SENTENCETRANSFORMERS:
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.CONFIGURATION.sentencetransformers.embeddingsModel.value,
                model_kwargs={"device": settings.CONFIGURATION.sentencetransformers.device.value},
                encode_kwargs={"normalize_embeddings": normalize}
            )
        elif embeddings_mode == EmbeddingsGenerationMode.INSTRUCTOR:
            embeddings = HuggingFaceInstructEmbeddings(
                model_name=settings.CONFIGURATION.instructor.embeddingsModel.value,
                model_kwargs={"device": settings.CONFIGURATION.instructor.device.value},
                encode_kwargs={"normalize_embeddings": normalize}
            )
        elif embeddings_mode == EmbeddingsGenerationMode.OPENAI:
            embeddings = OpenAIEmbeddings(
                model=settings.CONFIGURATION.openai.embeddingsModel.value,
                openai_api_key=settings.CONFIGURATION.openai.apikey
            )
        elif embeddings_mode == EmbeddingsGenerationMode.ELSER:
            embeddings = None
            retrieval_strategy = ElasticsearchStore.SparseVectorRetrievalStrategy()
        else:
            raise NotImplementedError
        
        store_kwargs = {
            "index_name": settings.CONFIGURATION.elasticsearch.index,
            "vector_query_field": "vector",
            "query_field": "text",
            "es_connection": self.__client,
            "distance_strategy": settings.CONFIGURATION.elasticsearch.similarity,
            "strategy": retrieval_strategy,
        }
        if embeddings is not None:
            store_kwargs["embedding"] = embeddings

        self.__store = ElasticsearchStore(**store_kwargs)

    def clear(self) -> None:
        """
        Destroys the data in the store.
        """
        self.__client.indices.delete(index=settings.CONFIGURATION.elasticsearch.index,
                                     ignore_unavailable=True)

    def index_chunks(self, doc_name: str, chunks: list[Document]) -> None:
        """
        Indexes chunks into the store.
        """
        logging.info("Indexing %d chunks from %s", len(chunks), doc_name)
        bulk_size = settings.CONFIGURATION.elasticsearch.bulkSize
        for i in range(0, len(chunks), bulk_size):
            logging.info("Batch %d-%d", i, i + bulk_size)
            self.__store.add_documents(chunks[i:i + bulk_size])
        logging.info("Indexed chunks from %s", doc_name)

    def extract_negations(self, query: str) -> (str, List[str]):
        """
        Extracts negations from the query and returns the modified query and negation filters.
        """
        negated_terms = re.findall(r'-\w+', query)
        negated_terms = [term[1:] for term in negated_terms]

        filters = []
        for term in negated_terms:
            filters.append(
                {
                    "bool": {
                        "must_not": [
                            {
                                "match": {
                                    "text": term
                                }
                            }
                        ]
                    }
                }
            )

        query = re.sub(r'-\w+', '', query)
        return query, filters

    def search(self, query: str) -> list[Document]:
        """
        Performs a vector search for the given query.
        """
        query, filters = self.extract_negations(query)
        results = self.__store.similarity_search_with_score(query,
                                                            K,
                                                            fetch_k=FETCH_K,
                                                            filter=filters)
        return results

    def ask(self, query: str) -> str:
        """
        Asks LLM for an answer based on the documents found by vector search
        """
        query, filters = self.extract_negations(query)
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=settings.CONFIGURATION.openai.apikey,
                model=settings.CONFIGURATION.openai.completionModel.value,
                temperature=settings.CONFIGURATION.openai.completionTemperature
            ),
            retriever=self.__store.as_retriever(search_kwargs={
                "k": K,
                "fetch_k": FETCH_K,
                "filter": filters
            }),
            chain_type_kwargs={"prompt": PROMPT}
        )
        return retrieval_qa.run(query)
