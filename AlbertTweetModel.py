from torch import nn
import torch
from transformers import AlbertConfig, AlbertModel

class AlbertTweetModel(nn.Module):
    def __init__(self):
        super(AlbertTweetModel, self).__init__()
        config = AlbertConfig.from_pretrained(
            './albert.torch/albert-base-v2/config.json',
            output_hidden_states=True
        )
        self.bert = AlbertModel.from_pretrained(
            './albert.torch/albert-base-v2/pytorch_model.bin',
            config=config
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.2)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Đầu vào Roberta cần chỉ số các token (input_ids)
        # Và attention_mask (Mặt nạ biểu diễn câu 0 = pad, 1 = otherwise)
        # Và token_type_ids
        _, _, hs = self.bert(input_ids, attention_mask, token_type_ids)

        # len(hs) = 13 tensor, mỗi tensor shape là (1, 128, 768)
        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        # x shape (4,1,128,768)
        x = torch.mean(x, 0)
        # x shape (1,128,768)
        x = self.dropout(x)
        x = self.fc(x)
        # x shape (1,128,2)
        start_logits, end_logits = x.split(1, dim=-1)

        # Nếu số chiều cuối là 1 thì bỏ đi (1,128,1) -> (1,128)
        # Ví dụ (AxBxCX1) --> size (AxBxC)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits