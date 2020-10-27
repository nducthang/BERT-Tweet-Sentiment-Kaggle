import torch
from AlbertTweetDataset import AlbertTweetDataset


def albert_get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    """Chia dữ liệu train và val

    Args:
        df (DataFrame): Dataframe dữ liệu
        train_idx (list): Danh sách chỉ số cho tập train
        val_idx (list): Danh sách chỉ số cho tập val
        batch_size (int, optional): Batchsize cho mô hình. Defaults to 8.
    """
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        AlbertTweetDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        AlbertTweetDataset(val_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    dataloader_dict = {"train": train_loader, "val": val_loader}

    return dataloader_dict
