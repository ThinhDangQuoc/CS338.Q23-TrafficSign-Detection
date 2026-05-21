"""Generate warning and explanation text from sign knowledge."""

from __future__ import annotations


def _value(sign_info: dict | None, key: str, fallback: str) -> str:
    value = (sign_info or {}).get(key)
    return str(value).strip() if value else fallback


def generate_warning(sign_info: dict) -> str:
    """Generate a short driver-facing warning."""
    short_name = _value(sign_info, "short_name", "biển báo giao thông")
    warning = _value(sign_info, "warning", "Hãy chú ý quan sát và điều chỉnh tốc độ phù hợp.")
    return f"Phát hiện biển {short_name}. {warning}"


def generate_full_explanation(sign_info: dict) -> str:
    """Generate a fuller learning explanation."""
    short_name = _value(sign_info, "short_name", "biển báo giao thông")
    sign_type = _value(sign_info, "type", "Chưa rõ loại biển")
    meaning = _value(sign_info, "meaning", "Chưa có ý nghĩa chi tiết.")
    action = _value(sign_info, "driver_action", "Hãy quan sát tình huống thực tế và tuân thủ quy định.")
    violation = _value(sign_info, "common_violation", "Chưa có dữ liệu lỗi thường gặp.")
    penalty = _value(sign_info, "penalty", "Mức phạt chỉ mang tính tham khảo và cần cập nhật.")
    example = _value(sign_info, "example", "Chưa có ví dụ minh họa.")
    return (
        f"{short_name} ({sign_type}).\n\n"
        f"Ý nghĩa: {meaning}\n\n"
        f"Người lái cần làm: {action}\n\n"
        f"Lỗi thường gặp: {violation}\n\n"
        f"Mức phạt tham khảo: {penalty}\n\n"
        f"Ví dụ: {example}"
    )


def generate_speech_text(sign_info: dict) -> str:
    """Generate concise text for text-to-speech."""
    short_name = _value(sign_info, "short_name", "biển báo giao thông")
    warning = _value(sign_info, "warning", "Hãy chú ý quan sát.")
    action = _value(sign_info, "driver_action", "Tuân thủ quy định giao thông.")
    return f"Phát hiện biển {short_name}. {warning} {action}"

