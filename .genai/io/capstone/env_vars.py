import os
from log import Log


class EnvVars:
    """
    Environment variable configuration.

    Supports multiple AI providers:
    - openai (ChatGPT) - default
    - gemini (Google Gemini)
    - claude (Anthropic Claude)

    Set AI_PROVIDER env var to choose provider.
    """

    # Supported AI providers
    PROVIDERS = {"openai", "gemini", "claude"}

    def __init__(self):
        # GitHub configuration
        self.owner = os.getenv('REPO_OWNER')
        self.repo = os.getenv('REPO_NAME')
        self.pull_number = os.getenv('PULL_NUMBER')
        self.token = os.getenv('GITHUB_TOKEN')
        self.base_ref = os.getenv('GITHUB_BASE_REF')
        self.head_ref = os.getenv('GITHUB_HEAD_REF')

        # AI Provider selection (default: openai)
        self.ai_provider = os.getenv('AI_PROVIDER', 'openai').lower()
        if self.ai_provider not in self.PROVIDERS:
            raise ValueError(
                f"Invalid AI_PROVIDER '{self.ai_provider}'. "
                f"Must be one of: {', '.join(self.PROVIDERS)}"
            )

        # AI API configuration based on provider
        if self.ai_provider == 'openai':
            self.ai_token = os.getenv('CHATGPT_KEY') or os.getenv('OPENAI_API_KEY')
            self.ai_model = os.getenv('CHATGPT_MODEL') or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        elif self.ai_provider == 'gemini':
            self.ai_token = os.getenv('GEMINI_KEY') or os.getenv('GOOGLE_API_KEY')
            self.ai_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        elif self.ai_provider == 'claude':
            self.ai_token = os.getenv('CLAUDE_KEY') or os.getenv('ANTHROPIC_API_KEY')
            self.ai_model = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')

        # Backwards compatibility
        self.chat_gpt_token = self.ai_token
        self.chat_gpt_model = self.ai_model

        # Target file extensions
        target_ext = os.getenv('TARGET_EXTENSIONS', '')
        if target_ext:
            self.target_extensions = [lang.strip() for lang in target_ext.split(",") if lang.strip()]
        else:
            self.target_extensions = []

        if len(self.target_extensions) == 0:
            raise ValueError(
                f"Please specify TARGET_EXTENSIONS. Comma separated, like: kt,java,py,js,swift,c,h. "
                f"Only these files will be reviewed"
            )

        # Required vars for validation
        self.env_vars = {
            "owner": self.owner,
            "repo": self.repo,
            "token": self.token,
            "base_ref": self.base_ref,
            "pull_number": self.pull_number,
            "ai_token": self.ai_token,
            "ai_model": self.ai_model,
        }

    def check_vars(self):
        missing_vars = [var for var, value in self.env_vars.items() if not value]
        if missing_vars:
            missing_vars_str = ", ".join(missing_vars)
            raise ValueError(f"The following environment variables are missing or empty: {missing_vars_str}")
        else:
            Log.print_green(f"All required environment variables are set. Using {self.ai_provider} provider.")
