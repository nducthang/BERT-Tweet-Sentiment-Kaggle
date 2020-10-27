import torch
from utils.compute_jaccard_score import compute_jaccard_score
import tqdm


def albert_train_model(model, dataloaders_dict, loss, optimizer, num_epochs, filename):
    model.cuda()

    for epoch in range(num_epochs):
        # Mỗi epoch sẽ thực hiện 2 phase
        for phase in ['train', 'val']:
            # Nếu phase train thì huấn luyện, phase val thì tính loss và jaccard
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Khởi tạo loss và jaccard
            epoch_loss = 0.0
            epoch_jaccard = 0.0

            for data in tqdm.tqdm((dataloaders_dict[phase])):
                # Lấy thông tin dữ liệu
                ids = data['ids'].cuda()
                masks = data['masks'].cuda()
                tweet = data['tweet']
                offsets = data['offsets'].numpy()
                token_type_id  = data['token_type_ids'].cuda()
                start_idx = data['start_idx'].cuda()
                end_idx = data['end_idx'].cuda()

                # Reset tích lũy đạo hàm
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    start_logits, end_logits = model(ids, masks, token_type_id)
                    loss_value = loss(
                        start_logits, end_logits, start_idx, end_idx)

                    # nếu là phase train thì thực hiện lan truyền ngược
                    # và cập nhật tham số
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                    epoch_loss += loss_value.item() * len(ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()

                    start_logits = torch.softmax(
                        start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(
                        end_logits, dim=1).cpu().detach().numpy()

                    # Tính toán jaccard cho tất cả các câu
                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i],
                            end_logits[i],
                            offsets[i]
                        )
                        epoch_jaccard += jaccard_score

            # Trung bình loss và jaccard
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / \
                len(dataloaders_dict[phase].dataset)

            print("Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(epoch +
                                                                                1, num_epochs, phase, epoch_loss, epoch_jaccard))
    torch.save(model.state_dict(), filename)
