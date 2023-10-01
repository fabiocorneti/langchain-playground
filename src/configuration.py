"""
Configuration classes
"""
import logging
from enum import Enum

from langchain.vectorstores.utils import DistanceStrategy
from pydantic import BaseModel


class LogLevel(Enum):
    """
    Logging level.
    """
    DEBUG = "debug"
    INFO = "info"


def get_logging_level(level: LogLevel) -> int:
    """
    Returns the logging level constant.
    """
    if level == LogLevel.DEBUG:
        return logging.DEBUG
    if level == LogLevel.INFO:
        return logging.INFO
    raise NotImplementedError


class EmbeddingsGenerationMode(Enum):
    """
    Embeddings generation mode.
    """
    INSTRUCTOR = "instructor"
    SENTENCETRANSFORMERS = "sentencetransformers"
    OPENAI = "openai"
    ELSER = "elser"


class PytorchDevice(Enum):
    """
    Supported pytorch devices
    """
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class InstructorEmbeddingsModel(Enum):
    """
    Supported instructor models
    """
    LARGE = "hkunlp/instructor-large"
    XL = "hkunlp/instructor-xl"


class OpenAICompletionModel(Enum):
    """
    Supported OpenAI completion models.
    """
    GPT35 = "gpt-3.5-turbo"
    GPT3516K = "gpt-3.5-turbo-16k"
    GPT4 = "gpt-4"
    GPT432K = "gpt-4-32k"


class OpenAIEmbeddingsModel(Enum):
    """
    Supported OpenAI embeddings models.
    """
    ADA002 = "text-embedding-ada-002"


class LargeLanguageModel(Enum):
    """
    Supported LLMs
    """
    NONE = "nope"
    OPENAI = "openai"


class SentenceTransformersEmbeddingsModel(Enum):
    """
    Supported embeddings models from HF's sentence-transformers
    """
    ALL_MINILM_L12_V2 = "all-MiniLM-L12-v2"
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE_V2 = "all-mpnet-base-v2"
    MSMARCO_DB_TASB = "msmarco-distilbert-base-tas-b"
    MSMARCO_DB_DOT_V5 = "msmarco-distilbert-dot-v5"


OPENAI_EMBEDDINGS_SIZES = {
    OpenAIEmbeddingsModel.ADA002: 1536
}


SENTENCETRANSFORMERS_EMBEDDINGS_SIZES = {
    SentenceTransformersEmbeddingsModel.ALL_MINILM_L12_V2: 384,
    SentenceTransformersEmbeddingsModel.ALL_MINILM_L6_V2: 384,
    SentenceTransformersEmbeddingsModel.ALL_MPNET_BASE_V2: 384,
    SentenceTransformersEmbeddingsModel.MSMARCO_DB_TASB: 768
}


class Splitter(Enum):
    """
    The text splitter.
    """
    RECURSIVE = "recursive"
    SPACY = "spacy"


class IndexerConfiguration(BaseModel):
    """
    Indexer configuration.
    """
    chunkSize: int = 800
    chunkOverlap: int = 200
    maxTokens: int = 500
    splitter: Splitter = Splitter


class ElasticsearchConfiguration(BaseModel):
    """
    Elasticsearch configuration.
    """
    url: str = "https://localhost:9200"
    username: str = "elastic"
    password: str = "changeme"
    index: str = "langchain"
    ignoreTlsVerification: bool = False
    similarity: DistanceStrategy = DistanceStrategy.COSINE
    requestTimeout: float = 60
    bulkSize: int = 100


class OpenAIConfiguration(BaseModel):
    """
    OpenAI configuration.
    """
    apikey: str
    embeddingsModel: OpenAIEmbeddingsModel = OpenAIEmbeddingsModel.ADA002
    completionModel: OpenAICompletionModel = OpenAICompletionModel.GPT3516K
    completionTemperature: int = 0


class SentenceTransformersConfiguration(BaseModel):
    """
    Sentence Transformers configuration.
    """
    embeddingsModel: SentenceTransformersEmbeddingsModel = \
        SentenceTransformersEmbeddingsModel.MSMARCO_DB_TASB
    device: PytorchDevice = PytorchDevice.CPU


class InstructorConfiguration(BaseModel):
    """
    Instructor configuration.
    """
    embeddingsModel: InstructorEmbeddingsModel = InstructorEmbeddingsModel.LARGE
    device: PytorchDevice = PytorchDevice.CPU


class Configuration(BaseModel):
    """
    Configuration.
    """
    elasticsearch: ElasticsearchConfiguration
    openai: OpenAIConfiguration
    sentencetransformers: SentenceTransformersConfiguration = SentenceTransformersConfiguration()
    instructor: InstructorConfiguration = InstructorConfiguration()
    embeddingsGenerationMode: EmbeddingsGenerationMode = \
        EmbeddingsGenerationMode.OPENAI
    indexer: IndexerConfiguration = IndexerConfiguration()
    llm: LargeLanguageModel = LargeLanguageModel.NONE
    loglevel: LogLevel = LogLevel.INFO

    def get_embedding_size(self) -> int:
        """
        Returns the text embedding size.
        """
        if self.embeddingsGenerationMode == EmbeddingsGenerationMode.SENTENCETRANSFORMERS:
            return SENTENCETRANSFORMERS_EMBEDDINGS_SIZES[self.sentencetransformers.embeddingsModel]
        if self.embeddingsGenerationMode == EmbeddingsGenerationMode.INSTRUCTOR:
            return 512
        return OPENAI_EMBEDDINGS_SIZES[self.openai.embeddingsModel]
