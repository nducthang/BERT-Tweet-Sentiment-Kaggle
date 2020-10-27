from sklearn.model_selection import StratifiedKFold
import pandas as pd
from RobertaTweetModel import RobertaTweetModel
import torch.optim as optim
from utils.loss import loss
from utils.roberta_get_train_val_loaders import roberta_get_train_val_loaders
from RobertaTrainModel import roberta_train_model
import torch

num_epochs = 3
batch_size = 1
seed_value = 28091997

torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
skl = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)

train_df = pd.read_csv('./data/train.csv')
train_df['text'] = train_df['text'].astype(str)
train_df['selected_text'] = train_df['selected_text'].astype(str)

for fold, (train_idx, val_idx) in enumerate(skl.split(train_df, train_df.sentiment), start=1):
    print("========== Fold {} ========== ".format(fold))
    model = RobertaTweetModel()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    loss = loss
    dataloader_dict = roberta_get_train_val_loaders(
        train_df, train_idx, val_idx, batch_size)
    roberta_train_model(
        model,
        dataloader_dict,
        loss,
        optimizer,
        num_epochs,
        f'./weights/roberta/roberta_fold_{fold}.bin'
    )
