import random
import numpy as np
import torch
from v3_dependecy_parser_comp import DependencyParser
from v3_utils_comp import generate_folds
from chu_liu_edmonds import decode_mst
import os


def train(fold_num, model, train_data_loader, validation_data_loader, epochs, lr, device):
    print(f'Beginning training, fold: {fold_num}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        model.train()
        sample_loss_lst = []
        for train_sentences_batch, train_labels_batch, train_positions_batch, train_real_seq_len_batch, train_d_tags_batch \
                in train_data_loader:
            batch_loss = 0
            optimizer.zero_grad()

            train_sentences_batch = train_sentences_batch.to(device)
            train_labels_batch = train_labels_batch.to(device)
            train_positions_batch = train_positions_batch.to(device)
            train_real_seq_len_batch = train_real_seq_len_batch.to(device)
            train_d_tags_batch = train_d_tags_batch.to(device)

            for sample_sentence, sample_y, sample_pos, sample_seq_len, sample_d_tags in zip(train_sentences_batch,
                                                                                            train_labels_batch,
                                                                                            train_positions_batch,
                                                                                            train_real_seq_len_batch,
                                                                                            train_d_tags_batch):
                sample_loss, sample_score_matrix = model(padded_sentence=sample_sentence,
                                                         padded_dependency_tree=sample_y,
                                                         padded_pos=sample_pos,
                                                         real_seq_len=sample_seq_len,
                                                         padded_d_tags=sample_d_tags)
                batch_loss = batch_loss + sample_loss
                sample_loss_lst.append(sample_loss.item())

            batch_loss.backward()
            optimizer.step()

        mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0], has_labels=False)
        uas_loss, val_loss = evaluate(model, validation_data_loader, device)
        print(
            f'Epoch: {i}, train loss: {np.average(sample_loss_lst)}, validation loss: {val_loss}, val uas: {uas_loss}')

    torch.save(model, f'models/comp_model_mlp_ex3_{fold_num}')


def evaluate(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        uas_loss_lst = []
        loss_lst = []
        total_tokens_num = 0
        # train_sentences_batch, train_labels_batch, train_positions_batch, train_real_seq_len_batch, train_d_tags_batch
        for x, y, pos, real_seq_len, d_tags in data_loader:
            x = torch.squeeze(x)[:real_seq_len].to(device)
            y = torch.squeeze(y)[:real_seq_len].to(device)
            d_tags = torch.squeeze(d_tags)[:real_seq_len].to(device)
            pos = torch.squeeze(pos)[:real_seq_len].to(device)
            real_seq_len = torch.squeeze(real_seq_len).to(device)
            loss, sample_score_matrix = model(padded_sentence=x,
                                              padded_dependency_tree=y,
                                              padded_pos=pos,
                                              padded_d_tags=d_tags,
                                              real_seq_len=real_seq_len)
            loss_lst.append(loss.item())
            mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0],
                                has_labels=False)
            uas_loss_lst += list((mst[1:] == y[1:real_seq_len].detach().cpu().numpy()))
            total_tokens_num += len(mst[1:])
            # uas_loss_lst.append(uas_loss)
    return sum(uas_loss_lst) / total_tokens_num, np.average(loss_lst)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def main(seed_num):
    set_seed(seed=seed_num)
    device = 'cuda'
    train_address = './data/train.labeled'
    val_address = './data/test.labeled'

    for idx, (train_data_loader, val_data_loader, sentences_word2idx, pos_word2idx) in enumerate(generate_folds(
            train_address=train_address,
            val_address=val_address,
            train_batch_size=24,
            train_shuffle=True,
            max_seq_len=250)):
        model = DependencyParser(device=device,
                                 embedding_dim=200,
                                 sentences_word2idx=sentences_word2idx,
                                 pos_word2idx=pos_word2idx).to(device)

        train(model=model,
              train_data_loader=train_data_loader,
              validation_data_loader=val_data_loader,
              epochs=1,
              lr=0.001,
              device=device,
              fold_num=idx)


if __name__ == '__main__':
    main(seed_num=130482)
