"""
Codebase indexer for building FAISS index of repository code.

Walks the repository, parses code files, chunks them at function/method level,
generates embeddings, and builds a searchable FAISS index with metadata.
"""

import os
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .embeddings import OpenAIEmbeddings
from .code_parser import CodeParser, CodeChunk, ImportType


@dataclass
class IndexedChunk:
    """Metadata for an indexed code chunk."""
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'import', 'block', 'file'
    name: Optional[str]  # Function/class name if applicable
    language: str
    content: str  # The actual code content


@dataclass
class CodebaseIndex:
    """Container for the FAISS index and associated metadata."""
    index: object  # faiss.Index
    chunks: List[IndexedChunk]
    import_graph: Dict[str, Set[str]]  # file -> set of imported files
    reverse_import_graph: Dict[str, Set[str]]  # file -> set of files that import it
    file_to_chunks: Dict[str, List[int]]  # file_path -> chunk indices


# Directories and files to skip during indexing
SKIP_DIRS = {
    '.git', '.svn', '.hg',
    'node_modules', 'vendor', 'venv', '.venv', 'env', '.env',
    '__pycache__', '.pytest_cache', '.mypy_cache',
    'build', 'dist', 'target', 'out',
    '.idea', '.vscode', '.vs',
    'coverage', '.coverage', 'htmlcov',
    'eggs', '*.egg-info',
}

SKIP_FILES = {
    '.DS_Store', 'Thumbs.db',
    'package-lock.json', 'yarn.lock', 'poetry.lock',
    '*.pyc', '*.pyo', '*.pyd',
    '*.so', '*.dll', '*.dylib',
    '*.min.js', '*.min.css',
}

# File extensions to index
INDEXABLE_EXTENSIONS = {
    # Python
    '.py',
    # JavaScript/TypeScript
    '.js', '.jsx', '.ts', '.tsx', '.mjs',
    # Java/Kotlin
    '.java', '.kt', '.kts',
    # Go
    '.go',
    # Rust
    '.rs',
    # C/C++
    '.c', '.cpp', '.cc', '.h', '.hpp',
    # Ruby
    '.rb',
    # PHP
    '.php',
    # C#
    '.cs',
    # Swift
    '.swift',
    # Documentation (for context)
    '.md', '.rst', '.txt',
}


class CodebaseIndexer:
    """
    Indexes a codebase for RAG retrieval.

    Chunks code at function/method level for optimal retrieval granularity:
    - Each function/method = 1 chunk
    - Each class = 1 chunk (or split if very large)
    - File header/imports = 1 chunk
    - Very large functions are split into sub-chunks

    This gives the best balance between:
    - Precision: Find the exact relevant function
    - Context: Entire function body provides complete understanding
    """

    # Maximum lines for a single chunk (large functions get split)
    MAX_CHUNK_LINES = 100

    def __init__(
        self,
        api_key: str,
        max_chunk_lines: int = 100
    ):
        """
        Initialize the indexer.

        Args:
            api_key: OpenAI API key for embeddings
            max_chunk_lines: Maximum lines per chunk (large functions get split)
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for indexing. "
                "Install with: pip install faiss-cpu"
            )

        self.embeddings = OpenAIEmbeddings(api_key)
        self.max_chunk_lines = max_chunk_lines
        # Parser configured to prefer function-level chunks
        self.parser = CodeParser(default_chunk_size=max_chunk_lines)

    def _should_skip_path(self, path: str) -> bool:
        """Check if a path should be skipped during indexing."""
        parts = path.split(os.sep)

        for part in parts:
            if part in SKIP_DIRS:
                return True
            for skip_pattern in SKIP_DIRS:
                if '*' in skip_pattern:
                    import fnmatch
                    if fnmatch.fnmatch(part, skip_pattern):
                        return True

        filename = os.path.basename(path)
        if filename in SKIP_FILES:
            return True
        for skip_pattern in SKIP_FILES:
            if '*' in skip_pattern:
                import fnmatch
                if fnmatch.fnmatch(filename, skip_pattern):
                    return True

        return False

    def _should_index_file(self, filepath: str) -> bool:
        """Check if a file should be indexed."""
        if self._should_skip_path(filepath):
            return False

        _, ext = os.path.splitext(filepath)
        return ext.lower() in INDEXABLE_EXTENSIONS

    def _collect_files(self, repo_path: str) -> List[str]:
        """Collect all indexable files in the repository."""
        files = []
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, repo_path)

                if self._should_index_file(rel_path):
                    files.append(rel_path)

        return files

    def _read_file_safe(self, repo_path: str, filepath: str) -> Optional[str]:
        """Safely read a file, handling encoding issues."""
        full_path = os.path.join(repo_path, filepath)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return None
        except Exception:
            return None

    def _build_import_graph(
        self,
        repo_path: str,
        files: List[str]
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Build import graph showing dependencies between files.

        Returns:
            Tuple of (import_graph, reverse_import_graph)
        """
        import_graph: Dict[str, Set[str]] = {}
        reverse_graph: Dict[str, Set[str]] = {}

        self.parser.set_repo_files(set(files))

        for filepath in files:
            content = self._read_file_safe(repo_path, filepath)
            if content is None:
                continue

            _, ext = os.path.splitext(filepath)
            language = self.parser.detect_language(ext)

            if language != 'python':
                continue

            local, _, _ = self.parser.get_imports_by_type(content, language)

            import_graph[filepath] = set()

            for imp in local:
                resolved = self._resolve_import(filepath, imp.full_path, files)
                if resolved:
                    import_graph[filepath].add(resolved)

                    if resolved not in reverse_graph:
                        reverse_graph[resolved] = set()
                    reverse_graph[resolved].add(filepath)

        return import_graph, reverse_graph

    def _resolve_import(
        self,
        current_file: str,
        import_path: str,
        all_files: List[str]
    ) -> Optional[str]:
        """Try to resolve an import path to an actual file."""
        path_from_import = import_path.replace('.', os.sep)
        candidates = [
            f"{path_from_import}.py",
            os.path.join(path_from_import, "__init__.py"),
        ]

        current_dir = os.path.dirname(current_file)
        if current_dir:
            candidates.extend([
                os.path.join(current_dir, f"{path_from_import}.py"),
                os.path.join(current_dir, path_from_import, "__init__.py"),
            ])

        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if candidate in all_files:
                return candidate

        return None

    def _chunk_file_by_functions(
        self,
        filepath: str,
        content: str,
        language: str
    ) -> List[IndexedChunk]:
        """
        Chunk a file at function/method level.

        Strategy:
        1. Extract all function/class definitions
        2. Create chunk for imports/header
        3. Create chunk for each function/method
        4. Split very large functions into sub-chunks
        """
        lines = content.split('\n')
        total_lines = len(lines)

        # For small files, keep as single chunk
        if total_lines <= 30:
            return [IndexedChunk(
                file_path=filepath,
                start_line=1,
                end_line=total_lines,
                chunk_type='file',
                name=os.path.basename(filepath),
                language=language,
                content=content
            )]

        # Get function/class definitions
        definitions = self.parser.extract_definitions(content, language)

        if not definitions:
            # No functions found - chunk as documentation or single block
            return [IndexedChunk(
                file_path=filepath,
                start_line=1,
                end_line=total_lines,
                chunk_type='file',
                name=os.path.basename(filepath),
                language=language,
                content=content
            )]

        chunks = []
        sorted_defs = sorted(definitions, key=lambda d: d['line'])

        # Chunk 1: Imports and file header (everything before first definition)
        first_def_line = sorted_defs[0]['line']
        if first_def_line > 1:
            header_content = '\n'.join(lines[:first_def_line - 1])
            if header_content.strip():
                chunks.append(IndexedChunk(
                    file_path=filepath,
                    start_line=1,
                    end_line=first_def_line - 1,
                    chunk_type='imports',
                    name='imports',
                    language=language,
                    content=header_content
                ))

        # Create chunk for each function/class
        for i, defn in enumerate(sorted_defs):
            start_line = defn['line']

            # End at next definition or EOF
            if i + 1 < len(sorted_defs):
                end_line = sorted_defs[i + 1]['line'] - 1
            else:
                end_line = total_lines

            chunk_lines = lines[start_line - 1:end_line]
            chunk_content = '\n'.join(chunk_lines)
            num_lines = len(chunk_lines)

            # If function is too large, split it
            if num_lines > self.max_chunk_lines:
                sub_chunks = self._split_large_function(
                    filepath=filepath,
                    lines=chunk_lines,
                    base_line=start_line,
                    defn=defn,
                    language=language
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(IndexedChunk(
                    file_path=filepath,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=defn['type'],
                    name=defn['name'],
                    language=language,
                    content=chunk_content
                ))

        return chunks

    def _split_large_function(
        self,
        filepath: str,
        lines: List[str],
        base_line: int,
        defn: Dict,
        language: str
    ) -> List[IndexedChunk]:
        """
        Split a large function into smaller chunks while preserving context.

        Keeps the function signature in each chunk for context.
        """
        chunks = []
        total_lines = len(lines)

        # Keep first few lines (signature + docstring) as context
        context_lines = min(10, total_lines // 3)
        context = '\n'.join(lines[:context_lines])

        # Split remaining lines
        remaining = lines[context_lines:]
        chunk_size = self.max_chunk_lines - context_lines
        overlap = 5

        i = 0
        part = 1
        while i < len(remaining):
            end_idx = min(i + chunk_size, len(remaining))
            chunk_lines = remaining[i:end_idx]

            # Prepend context (signature) for all chunks
            chunk_content = context + "\n# ... (continued)\n" + '\n'.join(chunk_lines)

            chunks.append(IndexedChunk(
                file_path=filepath,
                start_line=base_line + context_lines + i,
                end_line=base_line + context_lines + end_idx - 1,
                chunk_type=defn['type'],
                name=f"{defn['name']}_part{part}",
                language=language,
                content=chunk_content
            ))

            i += chunk_size - overlap
            part += 1

            if end_idx >= len(remaining):
                break

        return chunks

    def index_repository(
        self,
        repo_path: str = ".",
        show_progress: bool = True
    ) -> CodebaseIndex:
        """
        Index an entire repository at function/method level.

        Args:
            repo_path: Path to the repository root
            show_progress: Whether to print progress messages

        Returns:
            CodebaseIndex containing the FAISS index and metadata
        """
        repo_path = os.path.abspath(repo_path)

        if show_progress:
            print("Collecting files...")

        files = self._collect_files(repo_path)

        if show_progress:
            print(f"Found {len(files)} files to index")

        self.parser.set_repo_files(set(files))

        if show_progress:
            print("Building import graph...")

        import_graph, reverse_graph = self._build_import_graph(repo_path, files)

        if show_progress:
            print("Chunking files at function level...")

        all_chunks: List[IndexedChunk] = []
        file_to_chunks: Dict[str, List[int]] = {}
        chunk_texts: List[str] = []

        for filepath in files:
            content = self._read_file_safe(repo_path, filepath)
            if content is None or not content.strip():
                continue

            _, ext = os.path.splitext(filepath)
            language = self.parser.detect_language(ext)

            # Chunk at function level
            file_chunks = self._chunk_file_by_functions(filepath, content, language)

            file_chunk_indices = []

            for chunk in file_chunks:
                chunk_idx = len(all_chunks)
                file_chunk_indices.append(chunk_idx)
                all_chunks.append(chunk)

                # Prepare text for embedding with metadata
                embed_text = f"File: {filepath}\n"
                if chunk.name:
                    embed_text += f"{chunk.chunk_type}: {chunk.name}\n"
                embed_text += chunk.content
                chunk_texts.append(embed_text)

            if file_chunk_indices:
                file_to_chunks[filepath] = file_chunk_indices

        if show_progress:
            print(f"Created {len(all_chunks)} function-level chunks")
            print("Generating embeddings...")

        embeddings = self.embeddings.embed_batch(chunk_texts)

        if show_progress:
            print("Building FAISS index...")

        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        if show_progress:
            print(f"Index built with {index.ntotal} vectors")

        return CodebaseIndex(
            index=index,
            chunks=all_chunks,
            import_graph=import_graph,
            reverse_import_graph=reverse_graph,
            file_to_chunks=file_to_chunks
        )

    def index_single_file(
        self,
        filepath: str,
        content: str
    ) -> Tuple[List[IndexedChunk], np.ndarray]:
        """
        Index a single file (useful for incremental updates).

        Args:
            filepath: Path to the file
            content: File content

        Returns:
            Tuple of (chunks, embeddings_array)
        """
        _, ext = os.path.splitext(filepath)
        language = self.parser.detect_language(ext)

        file_chunks = self._chunk_file_by_functions(filepath, content, language)

        chunk_texts = []
        for chunk in file_chunks:
            embed_text = f"File: {filepath}\n"
            if chunk.name:
                embed_text += f"{chunk.chunk_type}: {chunk.name}\n"
            embed_text += chunk.content
            chunk_texts.append(embed_text)

        embeddings = self.embeddings.embed_batch(chunk_texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        return file_chunks, embeddings_array
