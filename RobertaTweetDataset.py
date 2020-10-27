import torch
import tokenizers
import pandas as pd

class RobertaTweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=128):
        # Dataframe dữ liệu
        self.df = df
        # độ dài tối đa của câu
        self.max_len = max_len
        # Nhãn
        self.labeled = 'selected_text' in df
        # Khởi tạo mã hóa BPE
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='./roberta.base.torch/vocab.json',
            merges_file='./roberta.base.torch/merges.txt',
            lowercase=True,
            add_prefix_space=True
        )

    def __len__(self):
        """ Trả về độ dài của DataFrame """
        return len(self.df)

    def get_input_data(self, row):
        """
        Tạo sample input cho 1 dòng dữ liệu
        - Input: <s><sentiment></s></s>token11 token12 ... </s><pad><pad>

        """
        # Thêm khoảng trắng vào đầu câu đầu vào
        tweet = " " + " ".join(row.text.lower().split())
        # Mã hóa BPE cho câu đầu vào
        encoding = self.tokenizer.encode(tweet)
        # {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        # 0 là đại diện cho token <s> và 2 là token </s>, 1 là token <pad>
        # Mã hóa câu đầu vào sang số
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        # offset là vị trí các token của câu ban đầu
        # Ví dụ (0,2) (2,3) (3,4) (4,9) ...
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        # Thêm các token pad cho viền câu
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0,0)] * pad_len

        ids = torch.tensor(ids)
        # Tạo mặt nạ, đánh dấu 1 cho toàn bộ câu đầu vào
        # Trừ các phần là <pad>
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        # Vị trí bắt đầu và kết thúc của selectec_text trong tweet
        idx0, idx1 = None, None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind:ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1

        # Đánh dấu những vị trí mà có ký tự của selected_text là 1
        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        # Đánh dấu những token chứa selected_text
        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        # Token bắt đầu và token kết thúc của selected_text
        start_idx = target_idx[0]
        end_idx = target_idx[-1]

        return start_idx, end_idx

    def __getitem__(self, index):
        """
        Chuyển đổi hàng dữ liệu thứ index trong dataFrame
        sang dữ liệu đầu vào của mô hình
        Các thuộc tính cho dữ liệu đầu vafo:
        - ids
        - masks
        - tweet
        - offsets
        - start_idx
        - end_idx
        """
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx

        return data
