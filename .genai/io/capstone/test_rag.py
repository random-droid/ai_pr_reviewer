#!/usr/bin/env python3
"""
Test script for RAG-powered PR reviewer.

Run this locally to verify the RAG system works before deploying.

Usage:
    # Test with Gemini (default - free tier)
    export GEMINI_KEY="your-google-api-key"
    python test_rag.py

    # Test with Gemini for both embeddings and AI
    export GEMINI_KEY="your-google-api-key"
    python test_rag.py --provider gemini

    # Test with Claude for AI (Gemini for embeddings)
    export GEMINI_KEY="your-google-api-key"
    export CLAUDE_KEY="your-anthropic-api-key"
    python test_rag.py --provider claude

    # Test with OpenAI for embeddings (requires paid API)
    export CHATGPT_KEY="your-openai-api-key"
    python test_rag.py --provider openai --embedding-provider openai
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all RAG modules can be imported."""
    print("=" * 60)
    print("TEST 1: Import RAG modules")
    print("=" * 60)

    try:
        from rag import (
            GeminiEmbeddings,
            OpenAIEmbeddings,
            CodeParser,
            CodebaseIndexer,
            RAGRetriever,
            StdlibDocsStore,
        )
        print("✓ All RAG modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nMake sure you have installed dependencies:")
        print("  pip install -r requirements.txt")
        return False


def test_code_parser():
    """Test code parsing and import classification."""
    print("\n" + "=" * 60)
    print("TEST 2: Code Parser")
    print("=" * 60)

    from rag import CodeParser, ImportType

    parser = CodeParser()

    # Test code with various imports
    test_code = '''
import os
import json
from datetime import datetime
from collections import defaultdict

import requests
import openai
from flask import Flask

from .utils import helper
from genai.chat_gpt import ChatGPT

def process_data(items):
    """Process a list of items."""
    result = defaultdict(list)
    for item in items:
        result[item.type].append(item)
    return result

class DataProcessor:
    def __init__(self):
        self.cache = {}

    def run(self):
        pass
'''

    # Test import extraction
    imports = parser.extract_imports(test_code, 'python')
    print(f"\nFound {len(imports)} imports:")

    stdlib_count = 0
    local_count = 0
    third_party_count = 0

    for imp in imports:
        icon = {
            ImportType.STDLIB: "📚",
            ImportType.LOCAL: "📁",
            ImportType.THIRD_PARTY: "📦"
        }.get(imp.import_type, "❓")
        print(f"  {icon} {imp.module_name} ({imp.import_type.value})")

        if imp.import_type == ImportType.STDLIB:
            stdlib_count += 1
        elif imp.import_type == ImportType.LOCAL:
            local_count += 1
        else:
            third_party_count += 1

    print(f"\nClassification: {stdlib_count} stdlib, {local_count} local, {third_party_count} third-party")

    # Test definition extraction
    definitions = parser.extract_definitions(test_code, 'python')
    print(f"\nFound {len(definitions)} definitions:")
    for defn in definitions:
        print(f"  - {defn['type']}: {defn['name']} (line {defn['line']})")

    # Test chunking
    chunks = parser.chunk_code(test_code, 'python', max_lines=20)
    print(f"\nChunked into {len(chunks)} chunks:")
    for chunk in chunks:
        name = chunk.name or "(unnamed)"
        print(f"  - {chunk.chunk_type}: {name} (lines {chunk.start_line}-{chunk.end_line})")

    print("\n✓ Code parser working correctly")
    return True


def test_stdlib_docs():
    """Test stdlib documentation store."""
    print("\n" + "=" * 60)
    print("TEST 3: Stdlib Documentation")
    print("=" * 60)

    from rag import StdlibDocsStore

    store = StdlibDocsStore()

    # Test getting docs by module
    os_docs = store.get_doc_by_module("os")
    print(f"\nFound {len(os_docs)} docs for 'os' module:")
    for doc in os_docs[:3]:
        print(f"  - {doc.name}")

    json_docs = store.get_doc_by_module("json")
    print(f"\nFound {len(json_docs)} docs for 'json' module:")
    for doc in json_docs[:3]:
        print(f"  - {doc.name}")

    # Test keyword search (no embeddings)
    results = store._keyword_search("file path join", top_k=3, module_filter=None)
    print(f"\nKeyword search for 'file path join':")
    for doc, score in results:
        print(f"  - {doc.name} (score: {score:.2f})")

    print("\n✓ Stdlib docs working correctly")
    return True


def test_indexer(api_key: str, embedding_provider: str = "gemini"):
    """Test codebase indexing."""
    print("\n" + "=" * 60)
    print("TEST 4: Codebase Indexer")
    print("=" * 60)

    from rag import CodebaseIndexer

    print(f"\nIndexing current directory with {embedding_provider.upper()} embeddings...")
    if embedding_provider == "gemini":
        print("(Using Gemini free tier)")
    else:
        print("(This will use OpenAI API credits)")

    indexer = CodebaseIndexer(api_key, embedding_provider=embedding_provider)
    index = indexer.index_repository(".", show_progress=True)

    print(f"\n✓ Indexed {len(index.chunks)} chunks")
    print(f"✓ Import graph has {len(index.import_graph)} files")
    print(f"✓ FAISS index has {index.index.ntotal} vectors")

    # Show some sample chunks
    print("\nSample indexed chunks:")
    for chunk in index.chunks[:5]:
        name = chunk.name or "(unnamed)"
        print(f"  - {chunk.file_path}: {chunk.chunk_type} '{name}'")

    return index


def test_retriever(api_key: str, index, embedding_provider: str = "gemini"):
    """Test RAG retrieval."""
    print("\n" + "=" * 60)
    print("TEST 5: RAG Retriever")
    print("=" * 60)

    from rag import RAGRetriever, GeminiEmbeddings, OpenAIEmbeddings, StdlibDocsStore

    if embedding_provider == "gemini":
        embeddings = GeminiEmbeddings(api_key)
    else:
        embeddings = OpenAIEmbeddings(api_key)

    stdlib_store = StdlibDocsStore(embeddings)
    stdlib_store.initialize()

    retriever = RAGRetriever(
        codebase_index=index,
        embeddings=embeddings,
        stdlib_store=stdlib_store
    )

    # Test retrieval for a sample file
    test_file = "github_reviewer.py"
    test_content = open(test_file).read() if os.path.exists(test_file) else "def test(): pass"
    test_diff = "+    result = process_data(items)\n+    return result"

    print(f"\nRetrieving context for: {test_file}")
    context = retriever.retrieve_context(
        file_path=test_file,
        file_content=test_content,
        diff_content=test_diff,
        language="python"
    )

    print(f"\nRetrieved context:")
    print(f"  - Imported files: {len(context.imported_files)}")
    print(f"  - Importing files: {len(context.importing_files)}")
    print(f"  - Related chunks: {len(context.related_chunks)}")
    print(f"  - Similar chunks: {len(context.similar_chunks)}")
    print(f"  - Stdlib docs: {len(context.stdlib_docs)}")

    # Show similar code
    if context.similar_chunks:
        print("\nMost similar code:")
        for chunk, score in context.similar_chunks[:3]:
            name = chunk.name or "(block)"
            print(f"  - {chunk.file_path}:{chunk.start_line} {name} (similarity: {score:.2f})")

    # Format for prompt
    formatted = context.format_for_prompt(max_tokens=1000)
    print(f"\nFormatted context length: {len(formatted)} chars")

    print("\n✓ Retriever working correctly")
    return True


def test_full_review(api_key: str, provider: str = "openai"):
    """Test a full review with AI."""
    print("\n" + "=" * 60)
    print(f"TEST 6: Full Review (with {provider.upper()})")
    print("=" * 60)

    from genai.ai_bot import AiBot

    # Get appropriate AI client
    if provider == "openai":
        from genai.chat_gpt import ChatGPT
        ai = ChatGPT(api_key, "gpt-4o-mini")
    elif provider == "gemini":
        from genai.gemini import Gemini
        ai = Gemini(api_key, "gemini-2.0-flash")
    elif provider == "claude":
        from genai.claude import Claude
        ai = Claude(api_key, "claude-sonnet-4-20250514")
    else:
        print(f"✗ Unknown provider: {provider}")
        return False

    # Sample code with intentional issues
    test_code = '''
import os
import json

def load_config(path):
    """Load configuration from file."""
    f = open(path, 'r')  # Issue: file not closed
    data = json.loads(f.read())
    return data

def process_user(user_id):
    """Process user data."""
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Issue: SQL injection
    return query

def calculate_average(numbers):
    """Calculate average of numbers."""
    total = sum(numbers)
    return total / len(numbers)  # Issue: division by zero if empty
'''

    test_diff = '''
+def load_config(path):
+    """Load configuration from file."""
+    f = open(path, 'r')  # Issue: file not closed
+    data = json.loads(f.read())
+    return data
+
+def process_user(user_id):
+    """Process user data."""
+    query = f"SELECT * FROM users WHERE id = {user_id}"  # Issue: SQL injection
+    return query
'''

    print("\nSending test code with intentional issues to AI...")
    print("Issues included:")
    print("  1. File not closed (resource leak)")
    print("  2. SQL injection vulnerability")
    print("  3. Division by zero risk")

    response = ai.ai_request_diffs(code=test_code, diffs=test_diff)

    print("\n" + "-" * 40)
    print("AI Response:")
    print("-" * 40)
    print(response)
    print("-" * 40)

    # Parse response
    comments = AiBot.split_ai_response(response)
    print(f"\nParsed {len(comments)} comments:")
    for comment in comments:
        text_preview = comment.text[:60] + "..." if len(comment.text) > 60 else comment.text
        print(f"  Line {comment.line}: {text_preview}")

    if AiBot.is_no_issues_text(response):
        print("\n⚠ AI found no issues (unexpected for this test)")
    else:
        print("\n✓ AI detected issues correctly")

    return True


def get_api_key(provider: str):
    """Get API key for the specified provider."""
    if provider == "openai":
        return os.environ.get('CHATGPT_KEY') or os.environ.get('OPENAI_API_KEY')
    elif provider == "gemini":
        return os.environ.get('GEMINI_KEY') or os.environ.get('GOOGLE_API_KEY')
    elif provider == "claude":
        return os.environ.get('CLAUDE_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    return None


def main():
    parser = argparse.ArgumentParser(description="Test RAG-powered PR reviewer")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "claude"],
        default="gemini",
        help="AI provider for reviews (default: gemini)"
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="Embedding provider (default: gemini - free tier)"
    )
    parser.add_argument(
        "--skip-indexer",
        action="store_true",
        help="Skip indexer/retriever tests"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RAG-POWERED PR REVIEWER - TEST SUITE")
    print("=" * 60)
    print(f"AI Provider: {args.provider.upper()}")
    print(f"Embedding Provider: {args.embedding_provider.upper()}")

    # Check API keys
    ai_api_key = get_api_key(args.provider)
    embedding_api_key = get_api_key(args.embedding_provider)

    if not ai_api_key:
        env_var = {
            "openai": "CHATGPT_KEY or OPENAI_API_KEY",
            "gemini": "GEMINI_KEY or GOOGLE_API_KEY",
            "claude": "CLAUDE_KEY or ANTHROPIC_API_KEY"
        }[args.provider]
        print(f"\n⚠ {env_var} environment variable not set")
        print("Set it to run AI review tests:")
        print(f"  export {env_var.split(' or ')[0]}='your-api-key'")

    if not embedding_api_key:
        env_var = {
            "openai": "CHATGPT_KEY or OPENAI_API_KEY",
            "gemini": "GEMINI_KEY or GOOGLE_API_KEY",
        }[args.embedding_provider]
        print(f"\n⚠ {env_var} environment variable not set for embeddings")
        print("Set it to run indexer tests:")
        print(f"  export {env_var.split(' or ')[0]}='your-api-key'")
        print("\nRunning offline tests only...\n")

    # Run tests
    tests_passed = 0
    tests_failed = 0

    # Test 1: Imports (no API needed)
    if test_imports():
        tests_passed += 1
    else:
        tests_failed += 1
        print("\n❌ Cannot continue without imports working")
        return False

    # Test 2: Code Parser (no API needed)
    if test_code_parser():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 3: Stdlib Docs (no API needed)
    if test_stdlib_docs():
        tests_passed += 1
    else:
        tests_failed += 1

    # Tests requiring embedding API key
    index = None
    if embedding_api_key and not args.skip_indexer:
        # Test 4: Indexer
        try:
            index = test_indexer(embedding_api_key, args.embedding_provider)
            if index:
                tests_passed += 1

                # Test 5: Retriever
                if test_retriever(embedding_api_key, index, args.embedding_provider):
                    tests_passed += 1
                else:
                    tests_failed += 1
            else:
                tests_failed += 1
        except Exception as e:
            print(f"\n✗ Indexer test failed: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    else:
        if args.skip_indexer:
            print("\n⏭ Skipping indexer tests (--skip-indexer)")
        else:
            print("\n⏭ Skipping indexer tests (need embedding API key)")

    # Test 6: Full Review (uses selected AI provider)
    if ai_api_key:
        try:
            if test_full_review(ai_api_key, args.provider):
                tests_passed += 1
            else:
                tests_failed += 1
        except Exception as e:
            print(f"\n✗ Full review test failed: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    else:
        print(f"\n⏭ Skipping AI review test (need {args.provider.upper()} API key)")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Passed: {tests_passed}")
    print(f"✗ Failed: {tests_failed}")

    if tests_failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed")

    return tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
