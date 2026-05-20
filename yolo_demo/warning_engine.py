"""Vietnamese warning messages for detected traffic signs."""

from __future__ import annotations

import unicodedata


DEFAULT_WARNING = "Phát hiện biển báo giao thông. Hãy chú ý quan sát và điều chỉnh tốc độ phù hợp."

WARNING_RULES = {
    "Duong nguoi di bo cat ngang": "Chú ý: Có vạch qua đường cho người đi bộ phía trước.",
    "Duong giao nhau (nga ba ben phai)": "Chú ý: Sắp đến ngã ba bên phải, cần giảm tốc độ.",
    "Cam di nguoc chieu": "Cảnh báo nguy hiểm: Cấm đi ngược chiều, không được tiếp tục đi vào.",
    "Phai di vong sang ben phai": "Chỉ dẫn: Phải đi vòng sang bên phải.",
    "Giao nhau voi duong dong cap": "Chú ý: Giao nhau với đường đồng cấp, cần quan sát hai phía.",
    "Giao nhau voi duong khong uu tien": "Chú ý: Sắp giao nhau với đường không ưu tiên.",
    "Cho ngoat nguy hiem vong ben trai": "Cảnh báo: Có chỗ ngoặt nguy hiểm về bên trái.",
    "Cam re trai": "Cảnh báo: Cấm rẽ trái tại khu vực này.",
    "Ben xe buyt": "Chú ý: Khu vực bến xe buýt, cần giảm tốc độ.",
    "Noi giao nhau chay theo vong xuyen": "Chú ý: Sắp đến vòng xuyến, đi theo hướng quy định.",
    "Cam dung va do xe": "Cảnh báo: Cấm dừng và đỗ xe.",
    "Cho quay xe": "Chỉ dẫn: Khu vực được phép quay xe.",
    "Bien gop lan duong theo phuong tien": "Chỉ dẫn: Chú ý phần làn đường theo từng loại phương tiện.",
    "Di cham": "Cảnh báo: Hãy đi chậm và giữ khoảng cách an toàn.",
    "Cam xe tai": "Cảnh báo: Cấm xe tải đi vào.",
    "Duong bi thu hep ve phia phai": "Chú ý: Đường bị thu hẹp về phía phải.",
    "Gioi han chieu cao": "Cảnh báo: Có giới hạn chiều cao, xe cao cần chú ý.",
    "Cam quay dau": "Cảnh báo: Cấm quay đầu xe.",
    "Cam o to khach va o to tai": "Cảnh báo: Cấm ô tô khách và ô tô tải.",
    "Cam re phai va quay dau": "Cảnh báo: Cấm rẽ phải và cấm quay đầu.",
    "Cam o to": "Cảnh báo: Cấm ô tô đi vào.",
    "Duong bi thu hep ve phia trai": "Chú ý: Đường bị thu hẹp về phía trái.",
    "Go giam toc phia truoc": "Chú ý: Có gờ giảm tốc phía trước.",
    "Cam xe hai va ba banh": "Cảnh báo: Cấm xe hai và ba bánh.",
    "Kiem tra": "Chú ý: Khu vực kiểm tra, cần tuân thủ hướng dẫn.",
    "Chi danh cho xe may*": "Chỉ dẫn: Làn đường chỉ dành cho xe máy.",
    "Chuong ngoai vat phia truoc": "Cảnh báo: Có chướng ngại vật phía trước.",
    "Tre em": "Cảnh báo: Khu vực có trẻ em, cần giảm tốc độ.",
    "Xe tai va xe cong*": "Chỉ dẫn: Làn đường dành cho xe tải và xe công.",
    "Cam mo to va xe may": "Cảnh báo: Cấm mô tô và xe máy.",
    "Chi danh cho xe tai*": "Chỉ dẫn: Làn đường chỉ dành cho xe tải.",
    "Duong co camera giam sat": "Chú ý: Khu vực có camera giám sát giao thông.",
    "Cam re phai": "Cảnh báo: Cấm rẽ phải tại khu vực này.",
    "Nhieu cho ngoat nguy hiem lien tiep, cho dau tien sang phai": "Cảnh báo: Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải.",
    "Cam xe so-mi ro-mooc": "Cảnh báo: Cấm xe sơ-mi rơ-moóc.",
    "Cam re trai va phai": "Cảnh báo: Cấm rẽ trái và rẽ phải.",
    "Cam di thang va re phai": "Cảnh báo: Cấm đi thẳng và rẽ phải.",
    "Duong giao nhau (nga ba ben trai)": "Chú ý: Sắp đến ngã ba bên trái, cần giảm tốc độ.",
    "Gioi han toc do (50km/h)": "Cảnh báo: Giới hạn tốc độ tối đa 50 km/h.",
    "Gioi han toc do (60km/h)": "Cảnh báo: Giới hạn tốc độ tối đa 60 km/h.",
    "Gioi han toc do (80km/h)": "Cảnh báo: Giới hạn tốc độ tối đa 80 km/h.",
    "Gioi han toc do (40km/h)": "Cảnh báo: Giới hạn tốc độ tối đa 40 km/h.",
    "Cac xe chi duoc re trai": "Chỉ dẫn: Các xe chỉ được rẽ trái.",
    "Chieu cao tinh khong thuc te": "Chú ý: Cần kiểm tra chiều cao tĩnh không phía trước.",
    "Nguy hiem khac": "Cảnh báo: Có nguy hiểm khác phía trước, cần giảm tốc độ.",
    "Duong mot chieu": "Chỉ dẫn: Đường một chiều.",
    "Cam do xe": "Cảnh báo: Cấm đỗ xe tại khu vực này.",
    "Cam o to quay dau xe (duoc re trai)": "Cảnh báo: Cấm ô tô quay đầu xe, được rẽ trái.",
    "Giao nhau voi duong sat co rao chan": "Cảnh báo: Giao nhau với đường sắt có rào chắn.",
    "Cam re trai va quay dau xe": "Cảnh báo: Cấm rẽ trái và quay đầu xe.",
    "Cho ngoat nguy hiem vong ben phai": "Cảnh báo: Có chỗ ngoặt nguy hiểm về bên phải.",
    "Chu y chuong ngai vat - vong tranh sang ben phai": "Cảnh báo: Có chướng ngại vật, vòng tránh sang bên phải.",
}


def _normalize(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d").replace("–", "-")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn").strip()


def get_warning(sign_name: str) -> str:
    normalized = _normalize(sign_name)
    if normalized in WARNING_RULES:
        return WARNING_RULES[normalized]

    lowered = normalized.lower()
    if "gioi han toc do" in lowered:
        return f"Cảnh báo: {sign_name}, hãy điều chỉnh tốc độ đúng quy định."
    if "cam" in lowered:
        return f"Cảnh báo: {sign_name}, cần tuân thủ lệnh cấm."
    if "nguy hiem" in lowered or "chuong" in lowered:
        return f"Cảnh báo: {sign_name}, hãy giảm tốc độ và quan sát."
    if "duong giao nhau" in lowered or "giao nhau" in lowered:
        return f"Chú ý: {sign_name}, cần quan sát trước khi đi tiếp."
    if "chi danh" in lowered or "chi duoc" in lowered:
        return f"Chỉ dẫn: {sign_name}."
    return DEFAULT_WARNING
