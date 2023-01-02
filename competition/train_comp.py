import random
import numpy as np
import torch
from competition.dependecy_parser_comp import DependencyParser
from competition.parser_comp import generate_ds
from chu_liu_edmonds import decode_mst
import os


def train(model, train_data_loader, validation_data_loader, epochs, lr, device):
    print('Beginning training')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        model.train()
        sample_loss_lst = []

        for train_data_batch, train_labels_batch, real_seq_len_batch in train_data_loader:
            batch_loss = 0
            optimizer.zero_grad()

            train_data_batch = train_data_batch.to(device)
            train_labels_batch = train_labels_batch.to(device)
            real_seq_len_batch = real_seq_len_batch.to(device)
            for sample_x, sample_y, sample_seq_len in zip(train_data_batch, train_labels_batch, real_seq_len_batch):
                sample_loss, sample_score_matrix = model(sample_x, sample_y, sample_seq_len)
                batch_loss = batch_loss + sample_loss
                sample_loss_lst.append(sample_loss.item())

            batch_loss.backward()
            optimizer.step()

        mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0], has_labels=False)
        uas_loss, val_loss = evaluate(model, validation_data_loader, device)
        print(
            f'Epoch: {i}, train loss: {np.average(sample_loss_lst)}, validation loss: {val_loss}, val uas: {uas_loss}')


def evaluate(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        uas_loss_lst = []
        loss_lst = []
        total_tokens_num = 0
        for x, y, real_seq_len in data_loader:
            x = torch.squeeze(x)[:real_seq_len].to(device)
            y = torch.squeeze(y)[:real_seq_len].to(device)
            real_seq_len = torch.squeeze(real_seq_len).to(device)
            loss, sample_score_matrix = model(x, y, real_seq_len)
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
    print(f"Random seed set as {seed}")


def main():
    set_seed(seed=318295029)
    device = 'cuda'
    train_file_address = '/home/user/PycharmProjects/nlp_ex_3/data/train.labeled'
    test_file_address = '/home/user/PycharmProjects/nlp_ex_3/data/test.labeled'
    train_data_loader = generate_ds(file_address=train_file_address,
                                    batch_size=25,
                                    shuffle=True)

    validation_data_loader = generate_ds(file_address=test_file_address,
                                         batch_size=1,
                                         shuffle=False)
    # Model initialization
    model = DependencyParser(device=device).to(device)

    train(model=model,
          train_data_loader=train_data_loader,
          validation_data_loader=validation_data_loader,
          epochs=50,
          lr=0.001,
          device=device)


if __name__ == '__main__':
    main()
