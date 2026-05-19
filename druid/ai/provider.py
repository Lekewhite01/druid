"""
druid.ai.provider
~~~~~~~~~~~~~~~~~

Abstract LLM provider interface and concrete implementations
for OpenAI, Anthropic (Claude), and Google (Gemini).

All providers expose a single ``complete()`` method that accepts
a system prompt + user message and returns the model's text response.
This keeps the rest of DRUID provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from druid.core.config import AIConfig


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    Parameters
    ----------
    config : AIConfig
        Provider configuration (API key, model, temperature, etc.).
    """

    def __init__(self, config: AIConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Parameters
        ----------
        user_message : str
            The main user prompt.
        system_prompt : str, optional
            System-level instructions.

        Returns
        -------
        str
            The model's response text.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""
        ...


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider (GPT-4o, GPT-4, etc.).

    Requires ``openai>=1.0.0``.  Install via::

        pip install druid-ai[openai]
    """

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install druid-ai[openai]"
            )

        client = OpenAI(api_key=self.config.api_key)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider (Claude Sonnet, Opus, Haiku).

    Requires ``anthropic>=0.30.0``.  Install via::

        pip install druid-ai[anthropic]
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def complete(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install druid-ai[anthropic]"
            )

        client = anthropic.Anthropic(api_key=self.config.api_key)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        # Extract text from content blocks
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        )


class GoogleProvider(LLMProvider):
    """
    Google Generative AI provider (Gemini).

    Requires ``google-generativeai>=0.5.0``.  Install via::

        pip install druid-ai[google]
    """

    @property
    def provider_name(self) -> str:
        return "google"

    def complete(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google provider requires google-generativeai. "
                "Install with: pip install druid-ai[google]"
            )

        genai.configure(api_key=self.config.api_key)

        model = genai.GenerativeModel(
            model_name=self.config.model,
            system_instruction=system_prompt,
        )

        response = model.generate_content(
            user_message,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
        )

        return response.text or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def get_provider(config: AIConfig) -> LLMProvider:
    """
    Instantiate the correct provider from an AIConfig.

    Parameters
    ----------
    config : AIConfig
        Must have ``provider`` set to one of
        ``"openai"``, ``"anthropic"``, or ``"google"``.

    Returns
    -------
    LLMProvider

    Raises
    ------
    ValueError
        If the provider is not supported.
    """
    cls = _PROVIDERS.get(config.provider)
    if cls is None:
        raise ValueError(
            f"Unsupported provider: {config.provider!r}. "
            f"Choose from: {list(_PROVIDERS.keys())}"
        )
    if not config.is_configured:
        raise ValueError(
            f"No API key found for provider {config.provider!r}. "
            f"Set it via DruidConfig or the corresponding environment variable."
        )
    return cls(config)
