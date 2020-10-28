import torch
import pandas as pd
import transformers
import sentencepiece as spm
from utils import sentencepiece_pb2


class OffsetTokenizer():
    def __init__(self, path_model):
        self.spt = sentencepiece_pb2.SentencePieceText()
        self.sp = spm.SentencePieceProcessor(model_file=path_model)

    def encode(self, text, lower=True):
        if lower:
            text = text.lower()
        offset = []
        ids = []
        self.spt.ParseFromString(self.sp.encode_as_serialized_proto(text))

        for piece in self.spt.pieces:
            offset.append((piece.begin, piece.end))
            ids.append(piece.id)

        return {"token": self.sp.encode_as_pieces(text), "ids": ids, "offsets": offset}


class AlbertTweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=128):
        # Dataframe dữ liệu
        self.df = df
        # Độ dài tối đa của câu
        self.max_len = max_len
        # Nhãn
        self.labeled = 'selected_text' in df
        # Khởi tạo mã hóa SentencePiece cho Albert
        self.tokenizer = OffsetTokenizer(
            path_model='./albert.torch/albert-base-v2/spiece.model')

        # self.sp = spm.SentencePieceProcessor(
        #     model_file='./albert.torch/albert-large-v2/spiece.model')
        # self.spt = sentencepiece_pb2.SentencePieceText()

        # ========= TEST CODE =======
        # self.tokenizer = transformers.AlbertTokenizer.from_pretrained(
        #     './albert.torch/albert-large-v2/spiece.model',
        #     do_lower_case=True)
        # print(self.tokenizer.tokenize(" Nguyen Duc Thang"))
        # print(self.tokenizer.encode("Nguyen Duc Thang"))
        # print(self.tokenizer.decode([2, 20449, 13, 8484, 119, 263, 3, 0]))
        # print(self.spt.ParseFromString(
        #     self.sp.encode_as_serialized_proto("Nguyen Duc Thang".lower())))
        # offset = []
        # ids = []

        # for piece in self.spt.pieces:
        #     offset.append((piece.begin, piece.end))
        #     ids.append(piece.id)
        # print(ids)
        # print(offset)
        # ======= END TEST =======

    def __len__(self):
        """ Trả về độ dài của DataFrame """
        return len(self.df)

    def get_input_data(self, row):
        """
        Tạo sample input cho 1 dòng dữ liệu
        - Input: [CLS] <sentiment>[SEP]token11 token12 ... [SEP][pad][pad]
        """
        # Thêm khoảng trắng vào đầu câu đầu vào
        tweet = " " + " ".join(row.text.lower().split())
        # Mã hóa câu đầu vào
        encoding = self.tokenizer.encode(tweet)
        # Mã hóa sentiment
        sentiment_id = self.tokenizer.encode(row.sentiment)["ids"]
        # 2 là CLS, 3 là SEP, 0 là <pad>
        ids = [2] + sentiment_id + [3] + encoding["ids"] + [3]
        # token type ids
        token_type_ids = [0] * 3 + [1] * (len(encoding["ids"]) + 1)
        # offset là vị trí các token của câu ban đầu
        offsets = [(0, 0)] * 3 + encoding["offsets"] + [(0, 0)]

        # Thêm các token pad cho viền
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
            offsets += [(0, 0)] * pad_len
            token_type_ids += [0] * pad_len
        ids = torch.tensor(ids)
        # Tạo mặt nạ attention, đánh dấu 1 cho toàn bộ câu đầu vào
        # Trừ các phần là <pad>
        masks = torch.where(ids != 0, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        token_type_ids = torch.tensor(token_type_ids)

        return ids, masks, tweet, offsets, token_type_ids

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
        - token_type_ids
        - start_idx
        - end_idx
        """
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets, token_type_ids = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        data['token_type_ids'] = token_type_ids

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx

        return data
