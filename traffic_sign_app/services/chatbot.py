"""Small rule-based chatbot for the selected/detected traffic sign."""

from __future__ import annotations

import unicodedata

from traffic_sign_app.core.warning_engine import generate_full_explanation


def _normalize(text: str) -> str:
    text = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def answer_question(question: str, sign_info: dict | None) -> str:
    """Answer a Vietnamese question using fields from the current sign."""
    if not sign_info:
        return "Bạn hãy nhận diện hoặc chọn một biển báo trước, rồi mình sẽ giải thích theo biển đó."

    q = _normalize(question or "")
    if not q.strip():
        return "Bạn có thể hỏi về ý nghĩa, cần làm gì, lỗi vi phạm, mức phạt tham khảo, ví dụ hoặc cảnh báo."

    if any(keyword in q for keyword in ["y nghia", "nghia la gi", "bien gi", "la gi"]):
        return sign_info.get("meaning", "Chưa có dữ liệu ý nghĩa cho biển này.")
    if any(keyword in q for keyword in ["lam gi", "can lam gi", "xu ly", "hanh dong"]):
        return sign_info.get("driver_action", "Chưa có dữ liệu hành động cần làm cho biển này.")
    if any(keyword in q for keyword in ["vi pham", "phat", "muc phat"]):
        violation = sign_info.get("common_violation", "Chưa có dữ liệu lỗi thường gặp.")
        penalty = sign_info.get("penalty", "Mức phạt chỉ mang tính tham khảo và cần cập nhật.")
        return f"Lỗi thường gặp: {violation}\n\nMức phạt tham khảo: {penalty}"
    if any(keyword in q for keyword in ["vi du", "tinh huong"]):
        return sign_info.get("example", "Chưa có ví dụ cho biển này.")
    if "canh bao" in q:
        return sign_info.get("warning", "Hãy chú ý quan sát và tuân thủ quy định.")

    return generate_full_explanation(sign_info)

