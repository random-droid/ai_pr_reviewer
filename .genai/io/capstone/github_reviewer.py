"""
GitHub PR Reviewer - Main orchestration script.

This script:
1. Builds a RAG index of the codebase for context
2. For each changed file in a PR:
   - Retrieves relevant context (related files, similar code, stdlib docs)
   - Sends to AI for review with full context
   - Posts review comments to the PR
"""

import os
from git import Git
from genai.ai_bot import AiBot
from log import Log
from env_vars import EnvVars
from repository.github import GitHub
from github_error import GitHubError

# AI Provider imports
def get_ai_client(provider: str, token: str, model: str):
    """
    Get the appropriate AI client based on provider.

    Args:
        provider: 'openai', 'gemini', or 'claude'
        token: API key
        model: Model name

    Returns:
        AI client instance
    """
    if provider == 'openai':
        from genai.chat_gpt import ChatGPT
        return ChatGPT(token, model)
    elif provider == 'gemini':
        from genai.gemini import Gemini
        return Gemini(token, model)
    elif provider == 'claude':
        from genai.claude import Claude
        return Claude(token, model)
    else:
        raise ValueError(f"Unknown AI provider: {provider}")


# RAG imports
try:
    from rag import (
        CodebaseIndexer,
        RAGRetriever,
        GeminiEmbeddings,
        OpenAIEmbeddings,
        StdlibDocsStore,
        CodeParser
    )
    RAG_AVAILABLE = True
except ImportError as e:
    Log.print_yellow(f"RAG module not available: {e}")
    Log.print_yellow("Running in legacy mode without codebase context")
    RAG_AVAILABLE = False


separator = "\n\n----------------------------------------------------------------------\n\n"
log_file = open('output.txt', 'a')


def build_rag_index(api_key: str, embedding_provider: str = "gemini"):
    """
    Build RAG index of the codebase.

    Args:
        api_key: API key for embeddings (Gemini or OpenAI)
        embedding_provider: "gemini" (default, free tier) or "openai"

    Returns:
        Tuple of (retriever, stdlib_store) or (None, None) if RAG unavailable
    """
    if not RAG_AVAILABLE:
        return None, None

    try:
        Log.print_green(f"Building RAG index of codebase (embeddings: {embedding_provider})...")

        # Initialize embeddings based on provider
        if embedding_provider == "gemini":
            embeddings = GeminiEmbeddings(api_key)
        else:
            embeddings = OpenAIEmbeddings(api_key)

        # Index the repository (uses same embedding provider)
        indexer = CodebaseIndexer(api_key, embedding_provider=embedding_provider)
        codebase_index = indexer.index_repository(".", show_progress=True)

        # Initialize stdlib docs
        stdlib_store = StdlibDocsStore(embeddings)
        stdlib_store.initialize()

        # Create retriever
        retriever = RAGRetriever(
            codebase_index=codebase_index,
            embeddings=embeddings,
            stdlib_store=stdlib_store
        )

        Log.print_green(f"RAG index built: {len(codebase_index.chunks)} chunks indexed")
        return retriever, stdlib_store

    except Exception as e:
        Log.print_red(f"Failed to build RAG index: {e}")
        Log.print_yellow("Falling back to legacy mode")
        return None, None


def get_language(file_extension: str) -> str:
    """Map file extension to language name."""
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'jsx': 'javascript',
        'tsx': 'typescript',
        'java': 'java',
        'kt': 'kotlin',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'cs': 'csharp',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'c',
        'hpp': 'cpp',
    }
    return language_map.get(file_extension.lower(), 'unknown')


def main():
    vars_main = EnvVars()
    vars_main.check_vars()

    # Initialize AI client based on provider
    ai = get_ai_client(vars_main.ai_provider, vars_main.ai_token, vars_main.ai_model)
    github = GitHub(vars_main.token, vars_main.owner, vars_main.repo, vars_main.pull_number)

    # Build RAG index for codebase context
    # Default: Gemini embeddings (free tier), fallback to OpenAI
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'gemini')
    if embedding_provider == 'gemini':
        embedding_key = os.getenv('GEMINI_KEY') or os.getenv('GOOGLE_API_KEY') or vars_main.ai_token
    else:
        embedding_key = os.getenv('OPENAI_API_KEY') or os.getenv('CHATGPT_KEY') or vars_main.ai_token
    retriever, stdlib_store = build_rag_index(embedding_key, embedding_provider)
    use_rag = retriever is not None

    remote_name = Git.get_remote_name()

    Log.print_green("Remote is", remote_name)
    changed_files = Git.get_diff_files(
        remote_name=remote_name,
        head_ref=vars_main.head_ref,
        base_ref=vars_main.base_ref
    )
    Log.print_green("Found changes in files", changed_files)

    if len(changed_files) == 0:
        Log.print_red("No changes between branch")
        return

    for file in changed_files:
        Log.print_green("Checking file", file)

        _, file_extension = os.path.splitext(file)
        file_extension = file_extension.lstrip('.')

        if file_extension not in vars_main.target_extensions:
            Log.print_yellow(f"Skipping, unsupported extension {file_extension} file {file}")
            continue

        try:
            with open(file, 'r') as file_opened:
                file_content = file_opened.read()
        except FileNotFoundError:
            Log.print_yellow("File was removed. Continue.", file)
            continue

        if len(file_content) == 0:
            Log.print_red("File is empty")
            continue

        file_diffs = Git.get_diff_in_file(
            remote_name=remote_name,
            head_ref=vars_main.head_ref,
            base_ref=vars_main.base_ref,
            file_path=file
        )

        if len(file_diffs) == 0:
            Log.print_red("Diffs are empty")
            continue

        language = get_language(file_extension)

        # Get RAG context if available
        rag_context = ""
        if use_rag:
            try:
                Log.print_green("Retrieving codebase context...")
                context = retriever.retrieve_context(
                    file_path=file,
                    file_content=file_content,
                    diff_content=file_diffs,
                    language=language
                )
                rag_context = context.format_for_prompt(max_tokens=2000)
                Log.print_green(f"Retrieved context: {len(rag_context)} chars")
            except Exception as e:
                Log.print_yellow(f"Failed to get RAG context: {e}")

        # Request AI review
        Log.print_green(f"Asking AI. Content Len:{len(file_content)} Diff Len: {len(file_diffs)}")

        if rag_context:
            response = ai.ai_request_with_context(
                code=file_content,
                diffs=file_diffs,
                rag_context=rag_context,
                file_path=file
            )
        else:
            response = ai.ai_request_diffs(code=file_content, diffs=file_diffs)

        # Log the request/response
        log_file.write(f"{separator}FILE: {file}{separator}")
        if rag_context:
            log_file.write(f"RAG CONTEXT:\n{rag_context}{separator}")
        log_file.write(f"DIFFS:\n{file_diffs}{separator}")
        log_file.write(f"RESPONSE:\n{response}{separator}")

        # Process response
        if AiBot.is_no_issues_text(response):
            Log.print_green("File looks good. Continue", file)
        else:
            responses = AiBot.split_ai_response(response)
            if len(responses) == 0:
                Log.print_red("Responses were not parsed:", response[:200])

            for comment in responses:
                result = False
                if comment.line:
                    result = post_line_comment(
                        github=github,
                        file=file,
                        text=comment.text,
                        line=comment.line
                    )
                if not result:
                    result = post_general_comment(
                        github=github,
                        file=file,
                        text=comment.text
                    )
                if not result:
                    Log.print_red("Failed to post comment:", comment.text[:100])


def post_line_comment(github: GitHub, file: str, text: str, line: int) -> bool:
    """Post a comment to a specific line in the PR."""
    Log.print_green("Posting line", file, line, text[:50])
    try:
        git_response = github.post_comment_to_line(
            text=text,
            commit_id=Git.get_last_commit_sha(file=file),
            file_path=file,
            line=line,
        )
        Log.print_yellow("Posted", git_response)
        return True
    except GitHubError as e:
        Log.print_red("Failed line comment", e)
        return False


def post_general_comment(github: GitHub, file: str, text: str) -> bool:
    """Post a general comment to the PR."""
    Log.print_green("Posting general", file, text[:50])
    try:
        message = f"**{file}**\n\n{text}"
        git_response = github.post_comment_general(message)
        Log.print_yellow("Posted general", git_response)
        return True
    except GitHubError:
        Log.print_red("Failed general comment")
        return False


if __name__ == "__main__":
    main()
    log_file.close()
