"""
RAG Retriever for fetching relevant context from the codebase index.

Combines multiple retrieval strategies:
1. Semantic search via FAISS (using OpenAI embeddings for meaning, FAISS for fast lookup)
2. Import graph traversal (related files)
3. Stdlib documentation lookup

How it works:
- OpenAI Embeddings: Converts code/text → vectors that capture semantic meaning
- FAISS: Stores vectors and finds similar ones very fast (locally, no API calls)
- Together: "semantic search" - find code that means similar things, not just keyword matches
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .embeddings import OpenAIEmbeddings
from .indexer import CodebaseIndex, IndexedChunk
from .stdlib_docs import StdlibDocsStore, StdlibDoc
from .code_parser import CodeParser, ImportType


@dataclass
class RetrievedContext:
    """Context retrieved for a file being reviewed."""
    # Similar code chunks from the codebase (found via semantic search)
    similar_chunks: List[Tuple[IndexedChunk, float]] = field(default_factory=list)

    # Files that this file imports (from import graph)
    imported_files: List[str] = field(default_factory=list)

    # Files that import this file (from import graph)
    importing_files: List[str] = field(default_factory=list)

    # Chunks from related files (imports/imported-by)
    related_chunks: List[IndexedChunk] = field(default_factory=list)

    # Relevant stdlib documentation
    stdlib_docs: List[Tuple[StdlibDoc, float]] = field(default_factory=list)

    def format_for_prompt(self, max_tokens: int = 2000) -> str:
        """
        Format retrieved context for injection into AI prompt.

        Args:
            max_tokens: Approximate token budget for context

        Returns:
            Formatted context string
        """
        sections = []
        estimated_tokens = 0
        tokens_per_char = 0.25  # Rough estimate

        # Section 1: Related Files (from import graph)
        if self.imported_files or self.importing_files:
            related_section = "## Related Files\n"
            if self.imported_files:
                related_section += f"**Imports:** {', '.join(self.imported_files[:5])}\n"
            if self.importing_files:
                related_section += f"**Imported by:** {', '.join(self.importing_files[:5])}\n"
            sections.append(related_section)
            estimated_tokens += len(related_section) * tokens_per_char

        # Section 2: Related Code Chunks (from import graph)
        if self.related_chunks and estimated_tokens < max_tokens * 0.4:
            chunks_section = "## Related Code\n"
            for chunk in self.related_chunks[:3]:
                chunk_text = f"### {chunk.file_path}:{chunk.start_line} ({chunk.chunk_type}"
                if chunk.name:
                    chunk_text += f": {chunk.name}"
                chunk_text += ")\n```\n"

                # Truncate content if too long
                content = chunk.content
                if len(content) > 500:
                    content = content[:500] + "\n# ... (truncated)"
                chunk_text += content + "\n```\n"

                if estimated_tokens + len(chunk_text) * tokens_per_char > max_tokens * 0.6:
                    break

                chunks_section += chunk_text
                estimated_tokens += len(chunk_text) * tokens_per_char

            sections.append(chunks_section)

        # Section 3: Similar Code Patterns (from semantic search)
        if self.similar_chunks and estimated_tokens < max_tokens * 0.7:
            similar_section = "## Similar Code Patterns\n"
            for chunk, score in self.similar_chunks[:3]:
                if score < 0.5:  # Skip low-relevance matches
                    continue

                chunk_text = f"### {chunk.file_path}:{chunk.start_line}"
                if chunk.name:
                    chunk_text += f" ({chunk.name})"
                chunk_text += f" [similarity: {score:.2f}]\n```\n"

                content = chunk.content
                if len(content) > 400:
                    content = content[:400] + "\n# ... (truncated)"
                chunk_text += content + "\n```\n"

                if estimated_tokens + len(chunk_text) * tokens_per_char > max_tokens * 0.85:
                    break

                similar_section += chunk_text
                estimated_tokens += len(chunk_text) * tokens_per_char

            if len(similar_section) > 30:  # Has content beyond header
                sections.append(similar_section)

        # Section 4: Standard Library Documentation
        if self.stdlib_docs and estimated_tokens < max_tokens * 0.9:
            docs_section = "## Relevant Python Standard Library\n"
            for doc, score in self.stdlib_docs[:2]:
                doc_text = f"### {doc.name}\n"
                doc_text += f"```python\n{doc.signature}\n```\n"
                doc_text += f"{doc.description}\n\n"
                doc_text += f"**Common pitfalls:** {doc.common_pitfalls}\n"

                if estimated_tokens + len(doc_text) * tokens_per_char > max_tokens:
                    break

                docs_section += doc_text
                estimated_tokens += len(doc_text) * tokens_per_char

            if len(docs_section) > 40:
                sections.append(docs_section)

        return "\n".join(sections) if sections else ""


class RAGRetriever:
    """
    Retrieves relevant context for code review using multiple strategies.

    Retrieval flow:
    1. Import Graph: Find directly related files (imports/imported-by)
    2. Semantic Search: Find similar code anywhere in codebase using embeddings
    3. Stdlib Docs: Find relevant Python standard library documentation
    """

    def __init__(
        self,
        codebase_index: CodebaseIndex,
        embeddings: OpenAIEmbeddings,
        stdlib_store: Optional[StdlibDocsStore] = None
    ):
        """
        Initialize the retriever.

        Args:
            codebase_index: Pre-built FAISS index with code chunks
            embeddings: OpenAI embeddings client (for query embedding)
            stdlib_store: Optional stdlib documentation store
        """
        self.index = codebase_index
        self.embeddings = embeddings
        self.stdlib_store = stdlib_store
        self.parser = CodeParser()

    def retrieve_context(
        self,
        file_path: str,
        file_content: str,
        diff_content: str,
        language: str = "python",
        top_k_similar: int = 5,
        top_k_stdlib: int = 3
    ) -> RetrievedContext:
        """
        Retrieve all relevant context for reviewing a file.

        Args:
            file_path: Path of the file being reviewed
            file_content: Full content of the file
            diff_content: Git diff for the file
            language: Programming language
            top_k_similar: Number of similar chunks to retrieve
            top_k_stdlib: Number of stdlib docs to retrieve

        Returns:
            RetrievedContext with all relevant information
        """
        context = RetrievedContext()

        # 1. Get related files from import graph (fast, no API calls)
        context.imported_files = list(
            self.index.import_graph.get(file_path, set())
        )
        context.importing_files = list(
            self.index.reverse_import_graph.get(file_path, set())
        )

        # 2. Get chunks from related files (fast, no API calls)
        related_file_paths = set(context.imported_files + context.importing_files)
        for related_path in list(related_file_paths)[:5]:  # Limit to 5 related files
            chunk_indices = self.index.file_to_chunks.get(related_path, [])
            for idx in chunk_indices[:2]:  # Get first 2 chunks per file
                if idx < len(self.index.chunks):
                    context.related_chunks.append(self.index.chunks[idx])

        # 3. Semantic search for similar code (1 API call for query embedding)
        context.similar_chunks = self._search_similar(
            query=diff_content or file_content,
            exclude_file=file_path,
            top_k=top_k_similar
        )

        # 4. Get relevant stdlib documentation (1 API call if using embeddings)
        if language == "python" and self.stdlib_store:
            context.stdlib_docs = self._get_relevant_stdlib(
                file_content=file_content,
                top_k=top_k_stdlib
            )

        return context

    def _search_similar(
        self,
        query: str,
        exclude_file: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[IndexedChunk, float]]:
        """
        Search for similar code chunks using FAISS.

        How it works:
        1. Convert query to embedding vector (OpenAI API call)
        2. FAISS finds nearest vectors in the index (local, fast)
        3. Return corresponding code chunks with similarity scores

        Args:
            query: Query text (code or diff)
            exclude_file: File to exclude from results (the file being reviewed)
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not query or self.index.index.ntotal == 0:
            return []

        # Step 1: Get query embedding (API call to OpenAI)
        query_embedding = self.embeddings.embed_text(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Step 2: Search FAISS index (local, very fast)
        search_k = top_k * 3 if exclude_file else top_k
        distances, indices = self.index.index.search(query_array, search_k)

        # Step 3: Convert results to chunks
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.index.chunks):
                continue

            chunk = self.index.chunks[idx]

            # Skip chunks from the file being reviewed
            if exclude_file and chunk.file_path == exclude_file:
                continue

            # Distance is inner product (cosine similarity for normalized vectors)
            similarity = float(dist)

            results.append((chunk, similarity))

            if len(results) >= top_k:
                break

        return results

    def _get_relevant_stdlib(
        self,
        file_content: str,
        top_k: int = 3
    ) -> List[Tuple[StdlibDoc, float]]:
        """
        Get relevant stdlib documentation based on imports in the file.

        Args:
            file_content: The file content to analyze
            top_k: Number of docs to return

        Returns:
            List of (StdlibDoc, relevance_score) tuples
        """
        if not self.stdlib_store:
            return []

        # Extract stdlib imports from the file
        _, stdlib_imports, _ = self.parser.get_imports_by_type(file_content, "python")

        if not stdlib_imports:
            return []

        # Get module names
        stdlib_modules = [imp.module_name for imp in stdlib_imports]

        # Search for relevant docs
        results = self.stdlib_store.search_relevant_docs(
            query=file_content[:2000],  # Limit query length
            top_k=top_k,
            module_filter=stdlib_modules if stdlib_modules else None
        )

        return results

    def search_by_function_name(
        self,
        function_name: str,
        top_k: int = 5
    ) -> List[Tuple[IndexedChunk, float]]:
        """
        Search for chunks by function/class name.

        Args:
            function_name: Name to search for
            top_k: Number of results

        Returns:
            List of matching chunks with scores
        """
        # First, try exact name match (no API call)
        exact_matches = []
        for chunk in self.index.chunks:
            if chunk.name and function_name.lower() in chunk.name.lower():
                exact_matches.append((chunk, 1.0))

        if exact_matches:
            return exact_matches[:top_k]

        # Fall back to semantic search (1 API call)
        return self._search_similar(
            query=f"function {function_name}",
            top_k=top_k
        )

    def get_file_context(
        self,
        file_path: str,
        include_related: bool = True
    ) -> List[IndexedChunk]:
        """
        Get all chunks for a specific file, optionally including related files.

        Args:
            file_path: Path to the file
            include_related: Whether to include chunks from related files

        Returns:
            List of relevant chunks
        """
        chunks = []

        # Get chunks for the file itself
        chunk_indices = self.index.file_to_chunks.get(file_path, [])
        for idx in chunk_indices:
            if idx < len(self.index.chunks):
                chunks.append(self.index.chunks[idx])

        if include_related:
            # Add chunks from imported files
            imported = self.index.import_graph.get(file_path, set())
            for imported_file in imported:
                imported_indices = self.index.file_to_chunks.get(imported_file, [])
                for idx in imported_indices[:2]:  # Limit per file
                    if idx < len(self.index.chunks):
                        chunks.append(self.index.chunks[idx])

        return chunks
