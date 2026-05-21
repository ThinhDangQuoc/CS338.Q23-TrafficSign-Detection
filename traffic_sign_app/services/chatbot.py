"""Small rule-based chatbot for the selected/detected traffic sign."""

from __future__ import annotations

import unicodedata

from traffic_sign_app.core.warning_engine import format_penalty_explanation, generate_full_explanation
from traffic_sign_app.services.knowledge_base import get_penalties_for_sign, get_speed_penalty


def _normalize(text: str) -> str:
    text = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _is_penalty_question(normalized_question: str) -> bool:
    return any(
        keyword in normalized_question
        for keyword in [
            "phat",
            "muc phat",
            "bao nhieu tien",
            "tru diem",
            "gplx",
            "can cu",
            "nghi dinh",
        ]
    )


def answer_question(
    question: str,
    sign_info: dict | None,
    vehicle_type: str | None = None,
    actual_speed: int | float | None = None,
) -> str:
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
    if _is_penalty_question(q):
        if sign_info.get("speed_limit_value") is not None:
            if actual_speed is None:
                return (
                    "Biển giới hạn tốc độ cần biết tốc độ thực tế của phương tiện để tính mức vượt quá. "
                    "Bạn hãy nhập tốc độ thực tế hoặc dùng phần tính trong giao diện."
                )
            penalty_record = get_speed_penalty(
                sign_info.get("speed_limit_value"),
                actual_speed,
                vehicle_type or "car",
            )
            return format_penalty_explanation(penalty_record, vehicle_type)

        penalty_records = get_penalties_for_sign(
            sign_info.get("class_id"),
            vehicle_type=vehicle_type or "car",
        )
        return format_penalty_explanation(penalty_records, vehicle_type)

    if "vi pham" in q:
        violation = sign_info.get("common_violation", "Chưa có dữ liệu lỗi thường gặp.")
        return f"Lỗi thường gặp: {violation}"
    if any(keyword in q for keyword in ["vi du", "tinh huong"]):
        return sign_info.get("example", "Chưa có ví dụ cho biển này.")
    if "canh bao" in q:
        return sign_info.get("warning", "Hãy chú ý quan sát và tuân thủ quy định.")

    return generate_full_explanation(sign_info)
