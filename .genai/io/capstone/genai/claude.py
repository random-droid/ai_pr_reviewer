"""
Anthropic Claude implementation for AI code review.
"""

import anthropic
from .ai_bot import AiBot


class Claude(AiBot):
    """Claude-based code reviewer."""

    def __init__(
        self,
        token: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize Claude client.

        Args:
            token: Anthropic API key
            model: Model to use (e.g., "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022")
        """
        self.__client = anthropic.Anthropic(api_key=token)
        self.__model = model

    def ai_request_diffs(self, code: str, diffs: str) -> str:
        """
        Request AI review WITHOUT RAG context.

        Args:
            code: Full file content
            diffs: Git diff output

        Returns:
            AI response with review comments
        """
        message = self.__client.messages.create(
            model=self.__model,
            max_tokens=4096,
            system=AiBot.get_system_message(),
            messages=[
                {
                    "role": "user",
                    "content": AiBot.build_ask_text(code=code, diffs=diffs)
                }
            ]
        )
        return message.content[0].text if message.content else ""

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
        message = self.__client.messages.create(
            model=self.__model,
            max_tokens=4096,
            system=AiBot.get_system_message(),
            messages=[
                {
                    "role": "user",
                    "content": AiBot.build_ask_text_with_context(
                        code=code,
                        diffs=diffs,
                        rag_context=rag_context,
                        file_path=file_path
                    )
                }
            ]
        )
        return message.content[0].text if message.content else ""
