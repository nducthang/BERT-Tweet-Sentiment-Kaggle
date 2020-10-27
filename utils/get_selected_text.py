def get_selected_text(text, start_idx, end_idx, offsets):
    """Trích xuất cụm từ được chọn

    Args:
        text (str): Văn bản đầu vào đầy đủ
        start_idx (int): Vị trí bắt đầu của token được chọn
        end_idx (int): Ví trí kết thúc của token được chọn
        offsets (list): Danh sách các offset token
    """
    selected_text = ""
    for ix in range(start_idx, end_idx+1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            selected_text += " "
    return selected_text
