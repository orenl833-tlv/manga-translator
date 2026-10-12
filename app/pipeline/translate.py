"""
Picks a translation provider based on config and re-exports the shared
data shapes, so the rest of the app (worker.py) only ever imports from
here and doesn't need to know which provider is configured.

Add a new provider by dropping in a translate_<name>.py with the same
`translate_batch(pages, target_lang) -> TranslationResult` signature and
adding one line below.
"""
from __future__ import annotations

from ..config import settings
from .translate_shared import PageInput, TranslationResult, check_full_pages_present  # re-exported

__all__ = ["PageInput", "TranslationResult", "translate_batch"]

_PROVIDERS = {
    "anthropic": "translate_anthropic",
    "openai": "translate_openai",
}


def translate_batch(pages: list[PageInput], target_lang: str) -> TranslationResult:
    provider = settings.translation_provider
    module_name = _PROVIDERS.get(provider)
    if module_name is None:
        raise RuntimeError(
            f"Unknown TRANSLATION_PROVIDER={provider!r} — expected one of {sorted(_PROVIDERS)}"
        )
    import importlib
    module = importlib.import_module(f".{module_name}", package=__package__)
    result = module.translate_batch(pages, target_lang)
    # Checked once here rather than per-provider so it applies regardless of
    # which one is configured — see check_full_pages_present's docstring.
    check_full_pages_present(pages, result.translations)
    return result
