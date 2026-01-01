"""
Pre-embedded Python standard library documentation for RAG context.

Contains curated documentation snippets for commonly used stdlib modules
with their embeddings pre-computed for fast retrieval.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StdlibDoc:
    """Documentation entry for a stdlib module/function."""
    module: str           # e.g., "os"
    name: str             # e.g., "os.path.join"
    signature: str        # e.g., "os.path.join(path, *paths)"
    description: str      # Brief description
    common_pitfalls: str  # Common mistakes/issues
    example: str          # Code example


# Pre-defined documentation for popular Python stdlib modules
# Focus on functions/classes that commonly cause bugs when misused
PYTHON_STDLIB_DOCS: List[StdlibDoc] = [
    # os module
    StdlibDoc(
        module="os",
        name="os.path.join",
        signature="os.path.join(path, *paths)",
        description="Join path components intelligently. The return value is the concatenation of path and all members of *paths, with exactly one directory separator following each non-empty part, except the last.",
        common_pitfalls="If a component is an absolute path, all previous components are thrown away. Don't use string concatenation for paths - always use os.path.join. Watch out for trailing slashes.",
        example='os.path.join("/home", "user", "file.txt")  # Returns "/home/user/file.txt"'
    ),
    StdlibDoc(
        module="os",
        name="os.environ",
        signature="os.environ",
        description="A mapping object representing the string environment. Modifications to this mapping are reflected in the environment.",
        common_pitfalls="os.environ['KEY'] raises KeyError if KEY doesn't exist. Use os.environ.get('KEY') or os.getenv('KEY') for safe access. Environment variables are strings - don't forget to convert types.",
        example='api_key = os.environ.get("API_KEY", "default_value")'
    ),
    StdlibDoc(
        module="os",
        name="os.makedirs",
        signature="os.makedirs(name, mode=0o777, exist_ok=False)",
        description="Recursive directory creation. Creates all intermediate directories needed to create the leaf directory.",
        common_pitfalls="Without exist_ok=True, raises FileExistsError if directory exists. Mode is ignored on Windows. Race conditions possible in concurrent code.",
        example='os.makedirs("path/to/dir", exist_ok=True)  # Won\'t fail if exists'
    ),

    # json module
    StdlibDoc(
        module="json",
        name="json.loads",
        signature="json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)",
        description="Deserialize a JSON string to a Python object.",
        common_pitfalls="Raises json.JSONDecodeError on invalid JSON. None in Python becomes null in JSON. Watch for encoding issues with non-ASCII. Large numbers may lose precision.",
        example='data = json.loads(\'{"key": "value"}\')  # Returns {"key": "value"}'
    ),
    StdlibDoc(
        module="json",
        name="json.dumps",
        signature="json.dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)",
        description="Serialize obj to a JSON formatted string.",
        common_pitfalls="Non-serializable objects (datetime, set, custom classes) raise TypeError. Use default parameter or custom encoder. NaN/Infinity are not valid JSON - set allow_nan=False to catch.",
        example='json.dumps({"key": "value"}, indent=2)  # Pretty print'
    ),

    # datetime module
    StdlibDoc(
        module="datetime",
        name="datetime.datetime.now",
        signature="datetime.datetime.now(tz=None)",
        description="Return the current local date and time. If tz is None, returns naive datetime with local time.",
        common_pitfalls="Returns naive datetime by default (no timezone info). Use datetime.now(timezone.utc) for timezone-aware UTC time. Comparing naive and aware datetimes raises TypeError.",
        example='from datetime import datetime, timezone\ndt = datetime.now(timezone.utc)  # Timezone-aware UTC'
    ),
    StdlibDoc(
        module="datetime",
        name="datetime.datetime.strptime",
        signature="datetime.datetime.strptime(date_string, format)",
        description="Parse a string representing a date/time according to a format.",
        common_pitfalls="Raises ValueError if string doesn't match format exactly. Format codes are case-sensitive (%Y vs %y). No timezone parsing with %z before Python 3.2.",
        example='datetime.strptime("2024-01-15", "%Y-%m-%d")'
    ),
    StdlibDoc(
        module="datetime",
        name="datetime.timedelta",
        signature="datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)",
        description="Represents a duration, the difference between two dates or times.",
        common_pitfalls="No months/years parameters (variable length). Total seconds via .total_seconds(), not .seconds (which is remainder). Negative timedeltas are valid.",
        example='from datetime import timedelta\nexpiry = datetime.now() + timedelta(days=7)'
    ),

    # re module
    StdlibDoc(
        module="re",
        name="re.search",
        signature="re.search(pattern, string, flags=0)",
        description="Scan through string looking for the first location where pattern produces a match.",
        common_pitfalls="Returns None if no match (check before calling .group()). Use raw strings r'' for patterns. re.match only matches at start of string - use re.search for anywhere.",
        example='match = re.search(r"\\d+", "abc123")\nif match:\n    print(match.group())  # "123"'
    ),
    StdlibDoc(
        module="re",
        name="re.sub",
        signature="re.sub(pattern, repl, string, count=0, flags=0)",
        description="Return string with all occurrences of pattern replaced by repl.",
        common_pitfalls="Backslash in replacement needs escaping or use raw string. Use \\1, \\2 for backreferences. count=0 means replace all.",
        example='re.sub(r"\\s+", " ", "too   many   spaces")  # "too many spaces"'
    ),

    # collections module
    StdlibDoc(
        module="collections",
        name="collections.defaultdict",
        signature="collections.defaultdict(default_factory=None, /[, ...])",
        description="Dict subclass that calls a factory function to supply missing values.",
        common_pitfalls="default_factory is called without arguments. Accessing missing key adds it to dict. Use dict.get() if you don't want auto-creation.",
        example='from collections import defaultdict\ncounts = defaultdict(int)\ncounts["key"] += 1  # No KeyError'
    ),
    StdlibDoc(
        module="collections",
        name="collections.Counter",
        signature="collections.Counter([iterable-or-mapping])",
        description="Dict subclass for counting hashable objects. Elements are stored as keys and counts as values.",
        common_pitfalls="Missing keys return 0 (not KeyError). most_common() returns list of tuples. Negative counts are allowed.",
        example='from collections import Counter\ncounts = Counter("mississippi")  # Counter({\'i\': 4, \'s\': 4, ...})'
    ),

    # typing module
    StdlibDoc(
        module="typing",
        name="typing.Optional",
        signature="typing.Optional[X]",
        description="Equivalent to Union[X, None]. Indicates a value can be X or None.",
        common_pitfalls="Optional[X] is NOT the same as X with a default value. Use for nullable parameters. In Python 3.10+, use X | None instead.",
        example='def greet(name: Optional[str] = None) -> str:\n    return f"Hello, {name or \'World\'}"'
    ),
    StdlibDoc(
        module="typing",
        name="typing.List",
        signature="typing.List[T]",
        description="Generic version of list. Deprecated in Python 3.9+ in favor of list[T].",
        common_pitfalls="Use list[T] in Python 3.9+. For heterogeneous lists, use List[Union[...]] or List[Any]. list is mutable - consider Sequence for read-only.",
        example='def process(items: list[int]) -> int:  # Python 3.9+\n    return sum(items)'
    ),

    # logging module
    StdlibDoc(
        module="logging",
        name="logging.getLogger",
        signature="logging.getLogger(name=None)",
        description="Return a logger with the specified name, creating it if necessary. If name is None, return the root logger.",
        common_pitfalls="Use __name__ as logger name for module hierarchy. Configure root logger before getLogger calls. Don't configure in library code - let application configure.",
        example='import logging\nlogger = logging.getLogger(__name__)\nlogger.info("Message")'
    ),
    StdlibDoc(
        module="logging",
        name="logging.basicConfig",
        signature="logging.basicConfig(**kwargs)",
        description="Configure the root logger. Only has effect on first call.",
        common_pitfalls="Only first call has effect - subsequent calls ignored. Call before any logging. For libraries, DON'T call basicConfig - use NullHandler.",
        example='logging.basicConfig(\n    level=logging.INFO,\n    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"\n)'
    ),

    # pathlib module
    StdlibDoc(
        module="pathlib",
        name="pathlib.Path",
        signature="pathlib.Path(*pathsegments)",
        description="PurePath subclass representing a path to a file or directory with I/O operations.",
        common_pitfalls="Use / operator for joining (not +). .read_text() and .write_text() handle file opening. exists() follows symlinks - use is_file()/is_dir() for type checks.",
        example='from pathlib import Path\npath = Path("dir") / "file.txt"\nif path.exists():\n    content = path.read_text()'
    ),

    # subprocess module
    StdlibDoc(
        module="subprocess",
        name="subprocess.run",
        signature="subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, ...)",
        description="Run command with arguments, wait for completion, return CompletedProcess instance.",
        common_pitfalls="shell=True is a security risk with user input. Use list for args, not string. capture_output=True to get stdout/stderr. check=True to raise on non-zero exit.",
        example='result = subprocess.run(["ls", "-l"], capture_output=True, text=True, check=True)\nprint(result.stdout)'
    ),

    # asyncio module
    StdlibDoc(
        module="asyncio",
        name="asyncio.run",
        signature="asyncio.run(coro, *, debug=None)",
        description="Execute coroutine and return result. Creates new event loop each call.",
        common_pitfalls="Cannot be called when event loop is already running. Use asyncio.create_task() inside async code. Don't mix asyncio with threading without care.",
        example='async def main():\n    await asyncio.sleep(1)\n    return "done"\n\nresult = asyncio.run(main())'
    ),
    StdlibDoc(
        module="asyncio",
        name="asyncio.gather",
        signature="asyncio.gather(*aws, return_exceptions=False)",
        description="Run awaitables concurrently. Results returned in same order as input.",
        common_pitfalls="Without return_exceptions=True, first exception cancels others. Results order matches input order, not completion order. Don't forget await.",
        example='results = await asyncio.gather(\n    fetch_url(url1),\n    fetch_url(url2),\n    return_exceptions=True\n)'
    ),

    # itertools module
    StdlibDoc(
        module="itertools",
        name="itertools.chain",
        signature="itertools.chain(*iterables)",
        description="Make an iterator that returns elements from the first iterable until exhausted, then from the next.",
        common_pitfalls="Returns iterator, not list. chain.from_iterable() for single iterable of iterables. Original iterables are consumed.",
        example='from itertools import chain\nlist(chain([1, 2], [3, 4]))  # [1, 2, 3, 4]'
    ),
    StdlibDoc(
        module="itertools",
        name="itertools.groupby",
        signature="itertools.groupby(iterable, key=None)",
        description="Make an iterator that returns consecutive keys and groups from the iterable.",
        common_pitfalls="MUST sort by key first - only groups consecutive identical keys. Group iterator consumed after next key. Store groups immediately if needed.",
        example='from itertools import groupby\ndata = sorted(items, key=lambda x: x.type)\nfor key, group in groupby(data, key=lambda x: x.type):\n    items_list = list(group)  # Consume immediately!'
    ),

    # functools module
    StdlibDoc(
        module="functools",
        name="functools.lru_cache",
        signature="@functools.lru_cache(maxsize=128, typed=False)",
        description="Decorator to wrap function with memoizing callable that saves recent calls.",
        common_pitfalls="Arguments must be hashable. Mutable default arguments cause issues. Use cache_clear() to reset. typed=True distinguishes int/float.",
        example='@lru_cache(maxsize=100)\ndef fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'
    ),
    StdlibDoc(
        module="functools",
        name="functools.partial",
        signature="functools.partial(func, /, *args, **kwargs)",
        description="Return a new partial object which behaves like func called with args and kwargs.",
        common_pitfalls="Positional args are prepended, kwargs are merged. Use for callback functions with extra parameters. Can't change already-set kwargs.",
        example='from functools import partial\nbase_print = partial(print, sep=", ", end="\\n\\n")\nbase_print(1, 2, 3)  # 1, 2, 3\\n\\n'
    ),

    # contextlib module
    StdlibDoc(
        module="contextlib",
        name="contextlib.contextmanager",
        signature="@contextlib.contextmanager",
        description="Decorator to create a context manager from a generator function.",
        common_pitfalls="Generator must yield exactly once. Use try/finally for cleanup. Exceptions propagate to caller unless handled.",
        example='from contextlib import contextmanager\n\n@contextmanager\ndef managed_resource():\n    resource = acquire()\n    try:\n        yield resource\n    finally:\n        release(resource)'
    ),

    # dataclasses module
    StdlibDoc(
        module="dataclasses",
        name="dataclasses.dataclass",
        signature="@dataclasses.dataclass(*, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)",
        description="Decorator that generates __init__, __repr__, __eq__ and other methods for a class.",
        common_pitfalls="Mutable default values (list, dict) shared between instances - use field(default_factory=list). frozen=True for immutability. order=True requires eq=True.",
        example='from dataclasses import dataclass, field\n\n@dataclass\nclass User:\n    name: str\n    tags: list = field(default_factory=list)'
    ),

    # hashlib module
    StdlibDoc(
        module="hashlib",
        name="hashlib.sha256",
        signature="hashlib.sha256(data=b'')",
        description="Return a new sha256 hash object. Can update with data and get digest.",
        common_pitfalls="Input must be bytes, not string - use .encode(). hexdigest() returns string, digest() returns bytes. Hash objects can't be copied after digest().",
        example='import hashlib\nhash_obj = hashlib.sha256(b"data")\nhash_obj.update(b"more data")\nprint(hash_obj.hexdigest())'
    ),
]


class StdlibDocsStore:
    """
    Store for pre-embedded Python stdlib documentation.

    Handles loading/saving embeddings and searching for relevant docs.
    """

    def __init__(self, embeddings_client=None, cache_path: str = None):
        """
        Initialize the stdlib docs store.

        Args:
            embeddings_client: OpenAIEmbeddings instance for generating/querying embeddings
            cache_path: Path to cached embeddings file
        """
        self.embeddings_client = embeddings_client
        self.docs = PYTHON_STDLIB_DOCS
        self.cache_path = cache_path or os.path.join(
            os.path.dirname(__file__), "data", "python_stdlib_embeddings.pkl"
        )
        self._doc_embeddings: Optional[List[List[float]]] = None
        self._doc_texts: Optional[List[str]] = None

    def _doc_to_text(self, doc: StdlibDoc) -> str:
        """Convert a StdlibDoc to searchable text."""
        return f"""{doc.name}
{doc.signature}
{doc.description}
Common issues: {doc.common_pitfalls}
Example: {doc.example}"""

    def _load_cache(self) -> bool:
        """Load cached embeddings if available."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                    self._doc_embeddings = cache['embeddings']
                    self._doc_texts = cache['texts']
                    return True
            except Exception:
                pass
        return False

    def _save_cache(self):
        """Save embeddings to cache file."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self._doc_embeddings,
                'texts': self._doc_texts
            }, f)

    def initialize(self, force_rebuild: bool = False):
        """
        Initialize embeddings (load from cache or generate).

        Args:
            force_rebuild: If True, regenerate embeddings even if cache exists
        """
        if not force_rebuild and self._load_cache():
            return

        if self.embeddings_client is None:
            raise ValueError("embeddings_client required to generate embeddings")

        # Generate embeddings for all docs
        self._doc_texts = [self._doc_to_text(doc) for doc in self.docs]
        self._doc_embeddings = self.embeddings_client.embed_batch(self._doc_texts)

        # Cache for next time
        self._save_cache()

    def get_doc_by_module(self, module_name: str) -> List[StdlibDoc]:
        """Get all docs for a specific module."""
        return [doc for doc in self.docs if doc.module == module_name]

    def get_doc_by_name(self, name: str) -> Optional[StdlibDoc]:
        """Get doc for a specific function/class name."""
        for doc in self.docs:
            if doc.name == name:
                return doc
        return None

    def search_relevant_docs(
        self,
        query: str,
        top_k: int = 3,
        module_filter: List[str] = None
    ) -> List[Tuple[StdlibDoc, float]]:
        """
        Search for relevant stdlib docs using semantic similarity.

        Args:
            query: Search query (code snippet or description)
            top_k: Number of results to return
            module_filter: If provided, only search these modules

        Returns:
            List of (StdlibDoc, score) tuples, sorted by relevance
        """
        if self._doc_embeddings is None:
            self.initialize()

        if self.embeddings_client is None:
            # Fallback to keyword matching if no embeddings client
            return self._keyword_search(query, top_k, module_filter)

        # Get query embedding
        query_embedding = self.embeddings_client.embed_text(query)

        # Calculate cosine similarity with all docs
        results = []
        for i, (doc, doc_embedding) in enumerate(zip(self.docs, self._doc_embeddings)):
            # Apply module filter
            if module_filter and doc.module not in module_filter:
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            results.append((doc, similarity))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        module_filter: List[str] = None
    ) -> List[Tuple[StdlibDoc, float]]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for doc in self.docs:
            if module_filter and doc.module not in module_filter:
                continue

            # Simple word overlap scoring
            doc_text = self._doc_to_text(doc).lower()
            doc_words = set(doc_text.split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                score = overlap / len(query_words)
                results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def format_docs_for_context(self, docs: List[Tuple[StdlibDoc, float]]) -> str:
        """Format docs for injection into AI prompt."""
        if not docs:
            return ""

        sections = []
        for doc, score in docs:
            section = f"""### {doc.name}
```python
{doc.signature}
```
{doc.description}

**Common pitfalls:** {doc.common_pitfalls}

**Example:**
```python
{doc.example}
```"""
            sections.append(section)

        return "\n\n".join(sections)
