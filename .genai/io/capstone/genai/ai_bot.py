"""
AI Bot base class for code review.

Provides:
- System message defining reviewer expertise
- Enhanced prompts with RAG context support
- Response parsing for line-specific comments
"""

from abc import ABC, abstractmethod
from typing import Optional
from .line_comment import LineComment


class AiBot(ABC):
    """Abstract base class for AI-powered code review."""

    # Response indicator for no issues
    __no_response = "No critical issues found"

    # System message defining AI reviewer's role and expertise
    __system_message = """You are an expert code reviewer with deep knowledge of software engineering best practices, security vulnerabilities, and performance optimization.

Your role is to review code changes (git diffs) and provide actionable, accurate feedback.

CORE PRINCIPLES:
1. ACCURACY OVER QUANTITY: Only report genuine issues. False positives waste developer time.
2. CONTEXT MATTERS: Consider the codebase context provided before flagging issues.
3. BE SPECIFIC: Reference exact line numbers and explain the root cause clearly.
4. BE ACTIONABLE: Every issue should have a clear path to resolution.
5. SEVERITY APPROPRIATENESS: Reserve critical issues for actual bugs/security problems.

WHAT TO LOOK FOR:
- Critical: Security vulnerabilities, data leaks, crashes, infinite loops, race conditions
- Warning: Potential null/undefined issues, resource leaks, poor error handling, edge cases
- Suggestion: Code style, naming conventions, performance improvements, maintainability

WHAT TO IGNORE (reduce false positives):
- Stylistic preferences that are subjective
- Working code that follows project conventions (check the codebase context)
- Changes that are intentional based on context
- TODOs or comments that are informational
- Test code that intentionally triggers edge cases"""

    # Prompt template WITHOUT RAG context (backwards compatible)
    __prompt_simple = """Review the following code changes for errors, issues, potential crashes, or unhandled exceptions.

OUTPUT FORMAT:
Print each issue in this format: "line_number : cause and effect"
If there are no issues, just say "{no_response}".

Do not add introductory text. Start directly with the issues or "{no_response}".

=== GIT DIFF ===
{diffs}
"""

    # Prompt template WITH RAG context
    __prompt_with_context = """Review the following code changes for errors, issues, potential crashes, or unhandled exceptions.

Use the CODEBASE CONTEXT below to understand:
- How similar code is written elsewhere in this project
- What functions/classes the changed code depends on
- Common patterns and conventions used in this codebase
- Standard library APIs being used (and their common pitfalls)

OUTPUT FORMAT:
Print each issue in this format: "line_number : cause and effect"
If there are no issues, just say "{no_response}".

Do not add introductory text. Start directly with the issues or "{no_response}".

=== CODEBASE CONTEXT ===
{rag_context}

=== FILE BEING REVIEWED ===
{file_path}

=== FULL FILE CONTENT ===
{code}

=== GIT DIFF (focus your review here) ===
{diffs}
"""

    @abstractmethod
    def ai_request_diffs(self, code: str, diffs: str) -> str:
        """
        Request AI review for code changes.

        Args:
            code: Full file content
            diffs: Git diff output

        Returns:
            AI response with review comments
        """
        pass

    def ai_request_with_context(
        self,
        code: str,
        diffs: str,
        rag_context: str = "",
        file_path: str = ""
    ) -> str:
        """
        Request AI review with RAG context.

        Args:
            code: Full file content
            diffs: Git diff output
            rag_context: Formatted RAG context string
            file_path: Path to the file being reviewed

        Returns:
            AI response with review comments
        """
        # Default implementation - subclasses can override
        return self.ai_request_diffs(code, diffs)

    @staticmethod
    def get_system_message() -> str:
        """Get the system message for AI reviewer."""
        return AiBot.__system_message

    @staticmethod
    def build_ask_text(code: str, diffs: str) -> str:
        """
        Build prompt WITHOUT RAG context (backwards compatible).

        Args:
            code: Full file content (currently unused but kept for API compatibility)
            diffs: Git diff output

        Returns:
            Formatted prompt string
        """
        return AiBot.__prompt_simple.format(
            no_response=AiBot.__no_response,
            diffs=diffs,
        )

    @staticmethod
    def build_ask_text_with_context(
        code: str,
        diffs: str,
        rag_context: str,
        file_path: str = ""
    ) -> str:
        """
        Build prompt WITH RAG context.

        Args:
            code: Full file content
            diffs: Git diff output
            rag_context: Formatted context from RAG retrieval
            file_path: Path to file being reviewed

        Returns:
            Formatted prompt string with context
        """
        return AiBot.__prompt_with_context.format(
            no_response=AiBot.__no_response,
            code=code,
            diffs=diffs,
            rag_context=rag_context if rag_context else "(No additional context available)",
            file_path=file_path if file_path else "(unknown)",
        )

    @staticmethod
    def is_no_issues_text(source: str) -> bool:
        """Check if response indicates no issues found."""
        target = AiBot.__no_response.replace(" ", "").lower()
        source_no_spaces = source.replace(" ", "").lower()
        return target in source_no_spaces

    @staticmethod
    def split_ai_response(input_response: str) -> list[LineComment]:
        """
        Parse AI response into line-specific comments.

        Expected format: "line_number : description"

        Args:
            input_response: Raw AI response

        Returns:
            List of LineComment objects
        """
        if input_response is None or not input_response:
            return []

        lines = input_response.strip().split("\n")
        models = []

        for full_text in lines:
            full_text = full_text.strip()
            if len(full_text) == 0:
                continue

            # Skip the "no issues" response
            if AiBot.is_no_issues_text(full_text):
                continue

            # Extract line number from start of text
            number_str = ''
            for char in full_text:
                if char.isdigit():
                    number_str += char
                else:
                    break

            number = int(number_str) if number_str else 0
            models.append(LineComment(line=number, text=full_text))

        return models
