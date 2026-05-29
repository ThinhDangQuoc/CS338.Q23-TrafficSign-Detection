"""Dual-engine chatbot for traffic sign education and Q&A.

Supports high-performance GPT-4o-mini (RAG-augmented) with auto-fallback to rule-based engine.
"""

from __future__ import annotations

import os
import unicodedata
import requests
from pathlib import Path

from traffic_sign_app.core.warning_engine import format_penalty_explanation, generate_full_explanation
from traffic_sign_app.services.knowledge_base import get_penalties_for_sign, get_speed_penalty

# Check for .env file and load manually to avoid external dependency issues
ENV_PATH = Path(__file__).parents[2] / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")


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


def get_rule_based_answer(
    question: str,
    sign_info: dict,
    vehicle_type: str | None = None,
    actual_speed: int | float | None = None,
) -> str:
    """Fallback rule-based answering engine."""
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


def answer_question(
    question: str,
    sign_info: dict | None,
    vehicle_type: str | None = None,
    actual_speed: int | float | None = None,
) -> str:
    """Answer a Vietnamese question using OpenAI GPT model with local fallback."""
    if not sign_info:
        return "Bạn hãy nhận diện hoặc chọn một biển báo trước, rồi mình sẽ giải thích theo biển đó."

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
        except Exception:
            pass
    
    # If no API key is set, immediately use rule-based fallback
    if not api_key:
        return get_rule_based_answer(question, sign_info, vehicle_type, actual_speed)

    # Auto-route between standard OpenAI endpoint and GitHub Models endpoint
    if api_key.startswith("github_pat_"):
        api_url = "https://models.inference.ai.azure.com/chat/completions"
    else:
        api_url = "https://api.openai.com/v1/chat/completions"

    # Context Retrieval for RAG (forces GPT to match our local SQLite knowledge database)
    short_name = sign_info.get("short_name", "biển báo giao thông")
    meaning = sign_info.get("meaning", "Chưa có ý nghĩa chi tiết.")
    action = sign_info.get("driver_action", "Tuân thủ quy định giao thông.")
    warning = sign_info.get("warning", "Hãy chú ý quan sát.")
    violation = sign_info.get("common_violation", "Chưa có dữ liệu lỗi thường gặp.")
    example = sign_info.get("example", "Chưa có ví dụ.")
    
    # Resolve penalty info for context
    if sign_info.get("speed_limit_value") is not None and actual_speed is not None:
        penalty_record = get_speed_penalty(sign_info.get("speed_limit_value"), actual_speed, vehicle_type or "car")
        penalty_text = format_penalty_explanation(penalty_record, vehicle_type)
    else:
        penalty_records = get_penalties_for_sign(sign_info.get("class_id"), vehicle_type=vehicle_type or "car")
        penalty_text = format_penalty_explanation(penalty_records, vehicle_type)

    system_prompt = (
        "Bạn là một chuyên gia tư vấn luật giao thông đường bộ Việt Nam và hệ thống biển báo đường bộ quy chuẩn QCVN 41:2019/BGTVT.\n"
        "Hãy giải thích câu hỏi của học viên một cách chi tiết, dễ hiểu, thân thiện dựa trên ngữ cảnh biển báo và luật giao thông được cung cấp dưới đây.\n\n"
        f"--- NGỮ CẢNH BIỂN BÁO ĐANG HỌC ---\n"
        f"Tên biển báo: {short_name}\n"
        f"Ý nghĩa: {meaning}\n"
        f"Hành động người lái cần thực hiện: {action}\n"
        f"Lỗi vi phạm thường gặp: {violation}\n"
        f"Cảnh báo an toàn: {warning}\n"
        f"Ví dụ/Tình huống minh họa: {example}\n"
        f"Chi tiết mức xử phạt vi phạm hành chính (áp dụng Nghị định 168/2024/NĐ-CP mới nhất có hiệu lực từ 01/01/2025 cho phương tiện {vehicle_type or 'ô tô'}): \n{penalty_text}\n"
        "------------------------------------\n\n"
        "QUY TẮC TRẢ LỜI:\n"
        "1. Trả lời bằng tiếng Việt lịch sự, khoa học. Dùng markdown (in đậm, danh sách) để cấu trúc rõ ràng câu trả lời.\n"
        "2. Bám sát ngữ cảnh biển báo để giải thích chính xác theo luật Việt Nam. Khi nói về mức phạt hoặc căn cứ pháp lý, hãy sử dụng chính xác thông tin trong ngữ cảnh được cung cấp.\n"
        "3. Nếu học viên hỏi ngoài lề biển báo hiện tại, hãy trả lời ngắn gọn rồi khéo léo định hướng họ quay lại tìm hiểu biển báo đang học."
    )

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.5,
            "max_tokens": 800
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=8
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            # Check for insufficient quota (429) or other errors, fallback to rule-based
            err_msg = response.json().get("error", {}).get("message", "")
            rule_ans = get_rule_based_answer(question, sign_info, vehicle_type, actual_speed)
            if "insufficient_quota" in err_msg or response.status_code == 429:
                return (
                    f"{rule_ans}\n\n"
                    f"*(⚠️ Trợ lý AI đang tạm thời sử dụng dữ liệu cục bộ do khóa API của bạn đã hết hạn mức (Quota Exceeded). "
                    f"Vui lòng nạp tiền hoặc đổi khóa API khác trong file .env để tiếp tục trò chuyện với GPT-4o-mini!)*"
                )
            return rule_ans

    except Exception:
        # Catch any timeout/network disconnect exceptions and fallback
        return get_rule_based_answer(question, sign_info, vehicle_type, actual_speed)


def generate_quiz_with_llm(sign_info: dict, difficulty: str = "medium") -> dict | None:
    """Generate a quiz question via LLM for a given traffic sign.

    Returns a dict with keys: question, options (list[str]), answer (str), explanation (str).
    Returns None if API is unavailable; caller should fall back to static scenarios.

    Args:
        sign_info: Sign knowledge dict from SQLite/JSON.
        difficulty: "easy" | "medium" | "hard" — controls question complexity.
    """
    import json as _json

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    api_url = (
        "https://models.inference.ai.azure.com/chat/completions"
        if api_key.startswith("github_pat_")
        else "https://api.openai.com/v1/chat/completions"
    )

    short_name = sign_info.get("short_name") or sign_info.get("class_name", "biển báo giao thông")
    meaning    = sign_info.get("meaning", "")
    action     = sign_info.get("driver_action", "")
    violation  = sign_info.get("common_violation", "")
    warning    = sign_info.get("warning", "")
    example    = sign_info.get("example", "")
    sign_type  = sign_info.get("type", "")

    difficulty_guide = {
        "easy":   "Tạo câu hỏi nhận biết đơn giản về ý nghĩa biển báo, phù hợp người mới học.",
        "medium": "Tạo câu hỏi áp dụng thực tế về hành động cần làm khi gặp biển, độ khó trung bình.",
        "hard":   "Tạo câu hỏi phân tích tình huống phức tạp, kết hợp nhiều quy tắc giao thông, độ khó cao.",
    }

    system_prompt = (
        "Bạn là chuyên gia ra đề thi luật giao thông đường bộ Việt Nam.\n"
        "Nhiệm vụ: tạo MỘT câu hỏi trắc nghiệm 4 đáp án về biển báo được cung cấp.\n\n"
        f"Mức độ: {difficulty_guide.get(difficulty, difficulty_guide['medium'])}\n\n"
        "Yêu cầu định dạng output - CHỈ trả về JSON, không có text nào khác:\n"
        "{\n"
        '  "question": "Nội dung câu hỏi (có thể kèm tình huống ngắn)",\n'
        '  "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],\n'
        '  "answer": "Đáp án đúng (copy nguyên văn từ options)",\n'
        '  "explanation": "Giải thích tại sao đúng, dẫn chiếu ý nghĩa biển và hành động cần làm"\n'
        "}\n\n"
        "Quy tắc:\n"
        "- Chỉ 1 đáp án đúng, 3 đáp án nhiễu hợp lý (không quá dễ loại trừ).\n"
        "- Đáp án nhiễu phải liên quan đến luật giao thông, không vô nghĩa.\n"
        "- Câu hỏi và đáp án viết bằng tiếng Việt, ngắn gọn, rõ ràng.\n"
        "- field 'answer' phải là chuỗi GIỐNG HỆT một phần tử trong 'options'."
    )

    user_prompt = (
        f"Biển báo: {short_name}\n"
        f"Loại biển: {sign_type}\n"
        f"Ý nghĩa: {meaning}\n"
        f"Hành động người lái: {action}\n"
        f"Lỗi vi phạm thường gặp: {violation}\n"
        f"Cảnh báo: {warning}\n"
        f"Ví dụ tình huống: {example}\n\n"
        "Hãy tạo câu hỏi quiz theo đúng định dạng JSON yêu cầu."
    )

    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.8,   # slightly higher for creative variation
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=12)
        if resp.status_code != 200:
            return None

        raw = resp.json()["choices"][0]["message"]["content"]
        quiz = _json.loads(raw)

        # Validate required keys
        if not all(k in quiz for k in ("question", "options", "answer", "explanation")):
            return None
        if not isinstance(quiz["options"], list) or len(quiz["options"]) < 2:
            return None
        # Ensure answer is in options (fuzzy match fallback)
        if quiz["answer"] not in quiz["options"]:
            for opt in quiz["options"]:
                if quiz["answer"].strip() in opt or opt.strip() in quiz["answer"]:
                    quiz["answer"] = opt
                    break

        return quiz

    except Exception:
        return None
