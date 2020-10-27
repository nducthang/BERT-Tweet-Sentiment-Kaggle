from torch import nn


def loss(start_logits, end_logits, start_positions, end_positions):
    """Tính mất mát dựa trên Cross Entropy

    Args:
        start_logits (logits): Vị trí bắt đầu - dự đoán
        end_logits (logits): Vị trí kết thúc - dự đoán
        start_positions (int): Vị trí bắt đầu - thực tế
        end_positions (int): Ví trí kết thúc - thực tế
    """
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss
