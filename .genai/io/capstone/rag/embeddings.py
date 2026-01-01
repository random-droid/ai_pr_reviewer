"""
Embeddings wrappers for generating vector representations of code.

Supports:
- OpenAI (text-embedding-3-small)
- Gemini (gemini-embedding-001)
"""

from typing import List, Protocol
from abc import ABC, abstractmethod


class EmbeddingsBase(ABC):
    """Abstract base class for embeddings providers."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        pass

    @abstractmethod
    def truncate_to_tokens(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limit."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class GeminiEmbeddings(EmbeddingsBase):
    """Wrapper for Google's Gemini embedding API (free tier available)."""

    MODEL = "models/gemini-embedding-001"
    MAX_TOKENS = 2048  # Conservative limit for Gemini
    DIMENSION = 768  # Using 768 for efficiency (supports 768, 1536, 3072)
    BATCH_SIZE = 100  # Gemini batch limit

    def __init__(self, api_key: str, dimension: int = 768):
        """
        Initialize the Gemini embeddings client.

        Args:
            api_key: Google AI API key (from ai.google.dev)
            dimension: Output dimension (768, 1536, or 3072)
        """
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def count_tokens(self, text: str) -> int:
        """Approximate token count (Gemini doesn't expose tokenizer)."""
        # Rough approximation: ~4 chars per token
        return len(text) // 4

    def truncate_to_tokens(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limit."""
        if max_tokens is None:
            max_tokens = self.MAX_TOKENS

        # Approximate: 4 chars per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        text = self.truncate_to_tokens(text)

        result = self._genai.embed_content(
            model=self.MODEL,
            content=text,
            output_dimensionality=self._dimension
        )
        return result['embedding']

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        truncated_texts = [self.truncate_to_tokens(t) for t in texts]

        all_embeddings = []
        for i in range(0, len(truncated_texts), self.BATCH_SIZE):
            batch = truncated_texts[i:i + self.BATCH_SIZE]
            # Gemini embed_content supports batch via list
            result = self._genai.embed_content(
                model=self.MODEL,
                content=batch,
                output_dimensionality=self._dimension
            )
            all_embeddings.extend(result['embedding'])

        return all_embeddings


class OpenAIEmbeddings(EmbeddingsBase):
    """Wrapper for OpenAI's text embedding API."""

    MODEL = "text-embedding-3-small"
    MAX_TOKENS = 8191  # Max tokens for text-embedding-3-small
    BATCH_SIZE = 2048  # Max inputs per API call
    DIMENSION = 1536  # text-embedding-3-small dimension

    def __init__(self, api_key: str):
        """
        Initialize the embeddings client.

        Args:
            api_key: OpenAI API key
        """
        import tiktoken
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def dimension(self) -> int:
        return self.DIMENSION

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
