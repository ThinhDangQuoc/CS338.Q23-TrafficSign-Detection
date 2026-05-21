"""Generate warning and explanation text from sign knowledge."""

from __future__ import annotations

from traffic_sign_app.services.knowledge_base import get_vehicle_label


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
    if (sign_info or {}).get("penalty_refs"):
        penalty = "Xem tab/khung Mức phạt tham khảo bên dưới."
    example = _value(sign_info, "example", "Chưa có ví dụ minh họa.")
    return (
        f"{short_name} ({sign_type}).\n\n"
        f"Ý nghĩa: {meaning}\n\n"
        f"Người lái cần làm: {action}\n\n"
        f"Lỗi thường gặp: {violation}\n\n"
        f"Mức phạt tham khảo: {penalty}\n\n"
        f"Ví dụ: {example}"
    )


def _format_number(value) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _format_vehicle_record(vehicle_key: str, vehicle: dict) -> list[str]:
    vehicle_name = get_vehicle_label(vehicle_key) or vehicle.get("vehicle_name")
    lines = [
        f"**Loại phương tiện:** {vehicle_name}",
        f"**Mức phạt:** {vehicle.get('fine', 'Chưa có dữ liệu')}",
        f"**Trừ điểm GPLX:** {vehicle.get('license_points', 'Chưa có dữ liệu')}",
        f"**Căn cứ:** {vehicle.get('legal_basis', 'Chưa có dữ liệu')}",
    ]
    if vehicle.get("note"):
        lines.append(f"**Ghi chú phương tiện:** {vehicle['note']}")
    return lines


def format_penalty_explanation(penalty_records, vehicle_type: str | None = None) -> str:
    """
    Convert penalty data into readable Vietnamese text.

    Includes behavior, vehicle type, fine, license points, legal basis, and notes.
    """
    if not penalty_records:
        return "Chưa có dữ liệu mức phạt cho biển này."

    records = penalty_records if isinstance(penalty_records, list) else [penalty_records]
    sections: list[str] = []
    for record in records:
        message = record.get("message")
        vehicle = record.get("vehicle") or {}
        by_vehicle = record.get("by_vehicle") or {}

        lines: list[str] = []
        behavior = record.get("behavior")
        if behavior:
            lines.append(f"**Hành vi:** {behavior}")

        if "speed_limit" in record and "actual_speed" in record:
            lines.append(f"**Tốc độ giới hạn:** {_format_number(record.get('speed_limit'))} km/h")
            lines.append(f"**Tốc độ thực tế:** {_format_number(record.get('actual_speed'))} km/h")
            lines.append(f"**Vượt quá:** {_format_number(record.get('over_speed'))} km/h")

        vehicle_name = vehicle.get("vehicle_name") or record.get("vehicle_name")
        if vehicle_name:
            lines.append(f"**Loại phương tiện:** {vehicle_name}")
        elif vehicle_type:
            lines.append(f"**Loại phương tiện:** {get_vehicle_label(vehicle_type)}")

        if vehicle:
            lines.append(f"**Mức phạt:** {vehicle.get('fine', 'Chưa có dữ liệu')}")
            lines.append(f"**Trừ điểm GPLX:** {vehicle.get('license_points', 'Chưa có dữ liệu')}")
            lines.append(f"**Căn cứ:** {vehicle.get('legal_basis', 'Chưa có dữ liệu')}")
            if vehicle.get("note"):
                lines.append(f"**Ghi chú phương tiện:** {vehicle['note']}")
        elif by_vehicle:
            for vehicle_key, vehicle_record in by_vehicle.items():
                lines.append("\n\n".join(_format_vehicle_record(vehicle_key, vehicle_record)))
        elif message:
            lines.append(message)

        if record.get("legal_document"):
            lines.append(f"**Văn bản:** {record['legal_document']}")
        if record.get("effective_from"):
            lines.append(f"**Hiệu lực:** {record['effective_from']}")
        if record.get("verified_at"):
            lines.append(f"**Cập nhật/kiểm chứng:** {record['verified_at']}")
        if record.get("note"):
            lines.append(f"**Ghi chú:** {record['note']}")

        sections.append("\n\n".join(lines) if lines else "Chưa có dữ liệu mức phạt cho biển này.")

    return "\n\n---\n\n".join(sections)


def generate_speech_text(sign_info: dict) -> str:
    """Generate concise text for text-to-speech."""
    short_name = _value(sign_info, "short_name", "biển báo giao thông")
    warning = _value(sign_info, "warning", "Hãy chú ý quan sát.")
    action = _value(sign_info, "driver_action", "Tuân thủ quy định giao thông.")
    return f"Phát hiện biển {short_name}. {warning} {action}"
