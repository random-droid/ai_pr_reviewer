"""
ChatGPT implementation for AI code review.

Provides:
- System message support for defining reviewer role
- RAG context-aware review requests
- Lower temperature for consistent reviews
"""

from openai import OpenAI
from .ai_bot import AiBot


class ChatGPT(AiBot):
    """ChatGPT-based code reviewer."""

    def __init__(
        self,
        token: str,
        model: str,
        temperature: float = 0.1  # Low temperature for consistent reviews
    ):
        """
        Initialize ChatGPT client.

        Args:
            token: OpenAI API key
            model: Model to use (e.g., "gpt-4o-mini", "gpt-4o")
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.__chat_gpt_model = model
        self.__client = OpenAI(api_key=token)
        self.__temperature = temperature

    def ai_request_diffs(self, code: str, diffs: str) -> str:
        """
        Request AI review WITHOUT RAG context (backwards compatible).

        Args:
            code: Full file content
            diffs: Git diff output

        Returns:
            AI response with review comments
        """
        response = self.__client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": AiBot.get_system_message(),
                },
                {
                    "role": "user",
                    "content": AiBot.build_ask_text(code=code, diffs=diffs),
                }
            ],
            model=self.__chat_gpt_model,
            temperature=self.__temperature,
        )
        return response.choices[0].message.content or ""

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
        response = self.__client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": AiBot.get_system_message(),
                },
                {
                    "role": "user",
                    "content": AiBot.build_ask_text_with_context(
                        code=code,
                        diffs=diffs,
                        rag_context=rag_context,
                        file_path=file_path
                    ),
                }
            ],
            model=self.__chat_gpt_model,
            temperature=self.__temperature,
        )
        return response.choices[0].message.content or ""
