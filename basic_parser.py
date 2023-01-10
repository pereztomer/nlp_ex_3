import numpy as np
from torch.utils.data import DataLoader, Dataset
from gensim import downloader


def parse_train_file(file_address):
    with open(file_address, encoding='utf-8') as f:
        sentences = []  # Contains the final sentences without tags
        sentence_tags = []  # Contains the tags of each word in the sentences
        new_sentence = []
        new_sentence_tags = []
        for row in f:
            if row != '\n':
                token = row.split('\t')[1]
                token_counter = row.split('\t')[0]
                token_head = row.split('\t')[6]
                new_sentence.append(token)
                new_sentence_tags.append(int(token_head))
            else:
                new_sentence.insert(0, 'ROOT')
                new_sentence_tags.insert(0, np.iinfo(np.int32).max)
                sentences.append(new_sentence)
                sentence_tags.append(new_sentence_tags)
                new_sentence = []
                new_sentence_tags = []

    return sentences, sentence_tags


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


class CustomDataset(Dataset):
    def __init__(self, x, y, seq_len_vals):
        self.x = x
        self.y = y
        self.seq_len_values = seq_len_vals

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.seq_len_values[idx]


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


def generate_ds(file_address, batch_size, shuffle):
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
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
