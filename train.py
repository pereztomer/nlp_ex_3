import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from gensim import downloader
from dependecy_parser_optimized import DependencyParser
from parser import parse_train_file


def embed_sentence(sen, embedding):
    representation = []
    for word in sen:
        if word == 'PADDING':
            vec = np.zeros(200)
        # word = word.lower()
        elif word not in embedding.key_to_index:
            vec = np.zeros(200)
        else:
            vec = embedding[word]
        representation.append(vec)
    representation = np.asarray(representation, dtype=np.float32)
    return representation


class PaddingDataset(Dataset):
    def __init__(self, x, y, max_seq_len):
        self.x = x
        self.y = y
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if len(self.x[idx]) == self.max_seq_len:
            return np.array(self.x[idx]), np.array(self.y[idx], dtype=np.int32), len(self.y[idx])
        elif len(self.x[idx]) > self.max_seq_len:
            original_x = np.array(self.x[idx])
            original_y = np.array(self.y[idx], dtype=np.int32)
            return original_x[:self.max_seq_len], original_y[:self.max_seq_len], len(self.y[idx])
        elif len(self.x[idx]) < self.max_seq_len:
            x_padded_val = self.x[idx] + (self.max_seq_len - len(self.x[idx])) * ['PADDING']
            y_padded_val = self.y[idx] + (self.max_seq_len - len(self.x[idx])) * [np.iinfo(np.int32).max]
            return np.array(x_padded_val), np.array(y_padded_val, dtype=np.int32), len(self.y[idx])


class CustomDataset(Dataset):
    def __init__(self, x, y, seq_len_vals):
        self.x = x
        self.y = y
        self.seq_len_values = seq_len_vals

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.seq_len_values[idx]


def train(model, data_loader, epochs, lr, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        print(f'Epoch: {i}')
        epoch_loss_lst = []

        for train_data_batch, train_labels_batch, real_seq_len_batch in data_loader:
            batch_loss = 0
            optimizer.zero_grad()

            train_data_batch = train_data_batch.to(device)
            train_labels_batch = train_labels_batch.to(device)
            real_seq_len_batch = real_seq_len_batch.to(device)
            for sample_x, sample_y, sample_seq_len in zip(train_data_batch, train_labels_batch, real_seq_len_batch):
                sample_loss, batch_score_matrix = model(sample_x, sample_y, sample_seq_len)
                batch_loss = batch_loss + sample_loss

            batch_loss.backward()
            optimizer.step()

            epoch_loss_lst.append(batch_loss.item())

        print(f'Epoch: {i}, train loss: {np.average(epoch_loss_lst)}')


def main():
    file_address = '/home/user/PycharmProjects/nlp_ex_3/data/train.labeled'
    sentences, sentence_tags = parse_train_file(file_address)
    max_seq_len = 250
    temp_data = PaddingDataset(x=sentences, y=sentence_tags, max_seq_len=max_seq_len)

    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)

    embedded_ds_x = []
    padded_ds_y = []
    padded_ds_seq_len = []
    for sen, tags, seq_len in temp_data:
        sen_embeddings = embed_sentence(sen=sen, embedding=glove)
        embedded_ds_x.append(sen_embeddings)
        padded_ds_y.append(tags)
        padded_ds_seq_len.append(seq_len)

    data = CustomDataset(x=embedded_ds_x, y=padded_ds_y, seq_len_vals=padded_ds_seq_len)

    data_loader = DataLoader(dataset=data,
                             batch_size=10,
                             shuffle=False)

    device = 'cuda'

    # Model initialization
    model = DependencyParser(device=device).to(device)

    train(model=model,
          data_loader=data_loader,
          epochs=50,
          lr=0.001,
          device=device)


if __name__ == '__main__':
    main()
