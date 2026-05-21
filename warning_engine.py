"""Backward-compatible imports for warning generation."""

from traffic_sign_app.core.warning_engine import (
    format_penalty_explanation,
    generate_full_explanation,
    generate_speech_text,
    generate_warning,
)

__all__ = ["generate_warning", "generate_full_explanation", "generate_speech_text", "format_penalty_explanation"]
