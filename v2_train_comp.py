import random
import numpy as np
import torch
from v3_dependecy_parser_comp import DependencyParser
from v2_utils_comp import generate_ds
from chu_liu_edmonds import decode_mst
import os
import matplotlib.pyplot as plt


def train(model, train_data_loader, train_data_redundant, validation_data_loader, epochs, lr, device):
    print('Beginning training')
    train_loss_list = []
    train_uas_list = []

    val_loss_list = []
    val_uas_list = []
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
        train_uas_loss, train_loss = evaluate(model, train_data_redundant, device)

        train_loss_list.append(train_loss)
        train_uas_list.append(train_uas_loss)

        val_loss_list.append(val_loss)
        val_uas_list.append(uas_loss)
        print(
            f'Epoch: {i}, train loss: {np.average(sample_loss_lst)}, validation loss: {val_loss}, val uas: {uas_loss}')

    torch.save(model, 'comp_model_mlp_ex3')
    return train_loss_list, train_uas_list, val_loss_list, val_uas_list


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


def plot_graph(train_loss, val_loss, graph_type):
    plt.plot(train_loss, label=f'train {graph_type}')
    plt.plot(val_loss, label=f'validation {graph_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'basic model {graph_type} - train/val')
    plt.legend()
    plt.show()


def main():
    set_seed(seed=318295029)
    device = 'cuda'
    train_address = './data/train.labeled'
    val_address = './data/test.labeled'

    train_data_loader, val_data_loader, sentences_word2idx, pos_word2idx = generate_ds(train_address=train_address,
                                                                                       val_address=val_address,
                                                                                       train_batch_size=24,
                                                                                       train_shuffle=True,
                                                                                       max_seq_len=250)

    train_data_redundant, _, _, _ = generate_ds(train_address=train_address,
                                                val_address=val_address,
                                                train_batch_size=1,
                                                train_shuffle=False,
                                                max_seq_len=250)

    # Model initialization
    model = DependencyParser(device=device,
                             embedding_dim=200,
                             sentences_word2idx=sentences_word2idx,
                             pos_word2idx=pos_word2idx).to(device)

    train_loss_list, train_uas_list, val_loss_list, val_uas_list = train(model=model,
                                                                         train_data_loader=train_data_loader,
                                                                         train_data_redundant=train_data_redundant,
                                                                         validation_data_loader=val_data_loader,
                                                                         epochs=10,
                                                                         lr=0.001,
                                                                         device=device)
    # plot_graph(train_loss_list, val_loss_list, 'loss')
    # plot_graph(train_uas_list, val_uas_list, 'uas')


if __name__ == '__main__':
    main()
