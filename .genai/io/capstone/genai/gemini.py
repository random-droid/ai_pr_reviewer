"""
Google Gemini implementation for AI code review.
"""

import google.generativeai as genai
from .ai_bot import AiBot


class Gemini(AiBot):
    """Gemini-based code reviewer."""

    def __init__(
        self,
        token: str,
        model: str = "gemini-1.5-flash",
    ):
        """
        Initialize Gemini client.

        Args:
            token: Google AI API key
            model: Model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
        """
        genai.configure(api_key=token)
        self.__model = genai.GenerativeModel(
            model_name=model,
            system_instruction=AiBot.get_system_message()
        )

    def ai_request_diffs(self, code: str, diffs: str) -> str:
        """
        Request AI review WITHOUT RAG context.

        Args:
            code: Full file content
            diffs: Git diff output

        Returns:
            AI response with review comments
        """
        prompt = AiBot.build_ask_text(code=code, diffs=diffs)
        response = self.__model.generate_content(prompt)
        return response.text or ""

    def ai_request_with_context(
        self,
        code: str,
        diffs: str,
        rag_context: str = "",
        file_path: str = ""
    ) -> str:
        """
        Request AI review WITH RAG context.

        Args:
            code: Full file content
            diffs: Git diff output
            rag_context: Formatted context from RAG retrieval
            file_path: Path to the file being reviewed

        Returns:
            AI response with review comments
        """
        prompt = AiBot.build_ask_text_with_context(
            code=code,
            diffs=diffs,
            rag_context=rag_context,
            file_path=file_path
        )
        response = self.__model.generate_content(prompt)
        return response.text or ""
