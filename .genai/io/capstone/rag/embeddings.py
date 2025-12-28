"""
OpenAI Embeddings wrapper for generating vector representations of code.
"""

from typing import List
import tiktoken
from openai import OpenAI


class OpenAIEmbeddings:
    """Wrapper for OpenAI's text embedding API."""

    MODEL = "text-embedding-3-small"
    MAX_TOKENS = 8191  # Max tokens for text-embedding-3-small
    BATCH_SIZE = 2048  # Max inputs per API call

    def __init__(self, api_key: str):
        """
        Initialize the embeddings client.

        Args:
            api_key: OpenAI API key
        """
        self._client = OpenAI(api_key=api_key)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self._tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int = None) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Input text to truncate
            max_tokens: Maximum tokens (defaults to model max)

        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self.MAX_TOKENS

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self._tokenizer.decode(truncated_tokens)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Truncate if necessary
        text = self.truncate_to_tokens(text)

        response = self._client.embeddings.create(
            model=self.MODEL,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Truncate all texts
        truncated_texts = [self.truncate_to_tokens(t) for t in texts]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(truncated_texts), self.BATCH_SIZE):
            batch = truncated_texts[i:i + self.BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self.MODEL,
                input=batch
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings
