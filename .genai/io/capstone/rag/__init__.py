"""
RAG (Retrieval-Augmented Generation) module for AI PR Reviewer.

This module provides codebase indexing and context retrieval to enhance
AI code reviews with relevant context from the entire repository.

How RAG works:
1. Indexing: Walk repo → chunk code at function level → embed with OpenAI → store in FAISS
2. Retrieval: For each file being reviewed:
   - Find related files via import graph
   - Find similar code via semantic search (FAISS)
   - Get relevant stdlib docs
3. Augmentation: Inject context into AI prompt for better reviews
"""

from .embeddings import OpenAIEmbeddings
from .code_parser import CodeParser, CodeChunk, ImportInfo, ImportType
from .indexer import CodebaseIndexer, CodebaseIndex, IndexedChunk
from .retriever import RAGRetriever, RetrievedContext
from .stdlib_docs import StdlibDocsStore, StdlibDoc

__all__ = [
    # Embeddings
    "OpenAIEmbeddings",
    # Code parsing
    "CodeParser",
    "CodeChunk",
    "ImportInfo",
    "ImportType",
    # Indexing
    "CodebaseIndexer",
    "CodebaseIndex",
    "IndexedChunk",
    # Retrieval
    "RAGRetriever",
    "RetrievedContext",
    # Stdlib docs
    "StdlibDocsStore",
    "StdlibDoc",
]
