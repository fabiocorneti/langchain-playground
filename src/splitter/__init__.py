"""
Custom splitters.
"""
from typing import List, Any
import spacy
from langchain.text_splitter import TextSplitter


class SpacyLimitTextSplitter(TextSplitter):
    """Splits text using Spacy splitter and enforcing a token limit.
    """

    def __init__(self, token_limit: int, separator: str = "\n\n", **kwargs: Any) -> None:
        """Constructor."""
        super().__init__(**kwargs)
        pipeline = "en_core_web_sm"
        spacy.cli.download(pipeline)
        self._separator = separator
        self._tokenizer = spacy.load(pipeline, exclude=["ner", "tagger"])
        self._tokenizer.add_pipe("token_splitter", config={
            "min_length": token_limit,
            "split_length": token_limit
        })

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = (s.text for s in self._tokenizer(text).sents)
        return self._merge_splits(splits, self._separator)


__all__ = ["SpacyLimitTextSplitter"]
