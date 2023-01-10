import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from numpy.random import default_rng


def parse_train_file(file_address):
    with open(file_address, encoding='utf-8') as f:
        sentences = []  # Contains the final sentences without tags
        sentence_dependency_tags = []  # Contains the tags of each word in the sentences
        sentence_tags = []  # Contains the tags of each word in the sentences
        sentence_positions = []
        new_sentence = []
        new_sentence_tags = []
        new_sentence_dependency_tags = []
        new_sentence_pos = []
        sentences_real_len = []
        for row in f:
            if row != '\n':
                token = row.split('\t')[1]
                token_pos = row.split('\t')[3]
                token_head = row.split('\t')[6]
                dependency_label = row.split('\t')[7]
                new_sentence.append(token)
                new_sentence_pos.append(token_pos)
                new_sentence_tags.append(int(token_head))
                new_sentence_dependency_tags.append(dependency_label)
            else:
                new_sentence.insert(0, 'ROOT')
                new_sentence_tags.insert(0, np.iinfo(np.int32).max)
                new_sentence_pos.insert(0, 'init')
                new_sentence_dependency_tags.insert(0, 'init')

                sentences.append(new_sentence)
                sentence_tags.append(new_sentence_tags)
                sentence_positions.append(new_sentence_pos)
                sentences_real_len.append(len(new_sentence))
                sentence_dependency_tags.append(new_sentence_dependency_tags)
                new_sentence = []
                new_sentence_tags = []
                new_sentence_pos = []
                new_sentence_dependency_tags = []

    return sentences, sentence_positions, sentence_tags, sentence_dependency_tags, sentences_real_len


class CustomDatasetAug(Dataset):
    def __init__(self, sentences, positions, tags, d_tags, seq_len_vals):
        self.sentences = sentences
        self.positions = positions
        self.tags = tags
        self.seq_len_values = seq_len_vals
        self.d_tags = d_tags
        self.sampler = default_rng()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample_size = int(self.seq_len_values[idx] * 0.3)
        numbers = self.sampler.choice(self.seq_len_values[idx], size=sample_size, replace=False)
        out_sen = copy.copy(self.sentences[idx])
        out_sen[numbers] = 1
        # if idx == 14:
        #     print(f'ids: {idx}, numbers: {numbers}')
        return out_sen, self.tags[idx], self.positions[idx], self.seq_len_values[idx], self.d_tags[idx]


class CustomDatasetNoAug(Dataset):
    def __init__(self, sentences, positions, tags, d_tags, seq_len_vals):
        self.sentences = sentences
        self.positions = positions
        self.tags = tags
        self.seq_len_values = seq_len_vals
        self.d_tags = d_tags
        self.sampler = default_rng()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tags[idx], self.positions[idx], self.seq_len_values[idx], self.d_tags[idx]


def tokenize(x_train, x_val):
    word2idx = {"[PAD]": 0, "[UNK]": 1}
    idx2word = ["[PAD]", "[UNK]"]
    for sent in x_train:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([word2idx[word] for word in sent])
    for sent in x_val:
        final_list_test.append([word2idx[word] if word in word2idx else word2idx['[UNK]'] for word in sent])
    return final_list_train, final_list_test, word2idx, idx2word


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features


def generate_ds(train_data, val_data, train_batch_size, train_shuffle, max_seq_len):
    # unpacking the train and validation:
    # train:
    train_sentences = train_data['sentences']
    train_y = train_data['y']
    train_sentence_dependency_tags = train_data['d_tags']
    train_positions = train_data['positions']
    train_sentences_real_len = train_data['real_len']
    # validation:
    val_sentences = val_data['sentences']
    val_y = val_data['y']
    val_sentence_dependency_tags = val_data['d_tags']
    val_positions = val_data['positions']
    val_sentences_real_len = val_data['real_len']

    train_sentences_idx, val_sentences_idx, sentences_word2idx, sentences_idx2word = tokenize(train_sentences,
                                                                                              val_sentences)
    train_sentences_idx_padded = padding_(train_sentences_idx, max_seq_len)
    train_y_padded = padding_(train_y, max_seq_len)
    val_sentences_idx_padded = padding_(val_sentences_idx, max_seq_len)
    val_y_padded = padding_(val_y, max_seq_len)

    # tokenizing positions:
    train_pos_idx, val_pos_idx, pos_word2idx, pos_idx2word = tokenize(train_positions, val_positions)
    train_pos_idx_padded = padding_(train_pos_idx, max_seq_len)
    val_pos_idx_padded = padding_(val_pos_idx, max_seq_len)

    # tokenizing dependency_tags:
    train_d_tags_idx, val_d_tags_idx, d_tags_word2idx, d_tags_idx2word = tokenize(train_sentence_dependency_tags,
                                                                                  val_sentence_dependency_tags)
    train_d_tags_idx_padded = padding_(train_d_tags_idx, max_seq_len)
    val_d_tags_idx_padded = padding_(val_d_tags_idx, max_seq_len)

    train_ds = CustomDatasetAug(sentences=train_sentences_idx_padded,
                                tags=train_y_padded,
                                positions=train_pos_idx_padded,
                                d_tags=train_d_tags_idx_padded,
                                seq_len_vals=train_sentences_real_len)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle)

    val_ds = CustomDatasetNoAug(sentences=val_sentences_idx_padded,
                                tags=val_y_padded,
                                positions=val_pos_idx_padded,
                                d_tags=val_d_tags_idx_padded,
                                seq_len_vals=val_sentences_real_len)

    val_data_loader = DataLoader(dataset=val_ds,
                                 batch_size=1,
                                 shuffle=False)

    return train_data_loader, val_data_loader, sentences_word2idx, pos_word2idx


def generate_folds(train_address, val_address, train_batch_size, train_shuffle, max_seq_len):
    train_sentences, train_positions, train_y, train_sentence_dependency_tags, train_sentences_real_len = \
        parse_train_file(train_address)

    val_sentences, val_positions, val_y, val_sentence_dependency_tags, val_sentences_real_len = parse_train_file(
        val_address)

    sentences = train_sentences + val_sentences
    positions = train_positions + val_positions
    y = train_y + val_y
    sentence_dependency_tags = train_sentence_dependency_tags + val_sentence_dependency_tags
    sentences_real_len = train_sentences_real_len + val_sentences_real_len

    available_indexes = list(range(len(sentences)))
    chunk_size = len(available_indexes) // 5
    split_indexes = [available_indexes[i:i + chunk_size] for i in range(0, len(available_indexes), chunk_size)]

    for i in range(5):
        fold_train_idx = []
        for sp_i, sp in enumerate(split_indexes):
            if sp_i == i:
                continue
            else:
                fold_train_idx.extend(sp)

        fold_train_sentences = []
        fold_train_positions = []
        fold_train_y = []
        fold_train_sentence_dependency_tags = []
        fold_train_sentences_real_len = []

        fold_val_sentences = []
        fold_val_positions = []
        fold_val_y = []
        fold_val_sentence_dependency_tags = []
        fold_val_sentences_real_len = []
        for counter, (sen, pos, sa_y, d_tag, real_len) in enumerate(
                zip(sentences, positions, y, sentence_dependency_tags,
                    sentences_real_len)):
            if counter in fold_train_idx:
                fold_train_sentences.append(sen)
                fold_train_positions.append(pos)
                fold_train_y.append(sa_y)
                fold_train_sentence_dependency_tags.append(d_tag)
                fold_train_sentences_real_len.append(real_len)
            else:
                fold_val_sentences.append(sen)
                fold_val_positions.append(pos)
                fold_val_y.append(sa_y)
                fold_val_sentence_dependency_tags.append(d_tag)
                fold_val_sentences_real_len.append(real_len)

        # send fold train and validation into generate ds:

        # print(len(fold_val_sentences))
        train_data = {'sentences': fold_train_sentences,
                      'positions': fold_train_positions,
                      'y': fold_train_y,
                      'd_tags': fold_train_sentence_dependency_tags,
                      'real_len': fold_train_sentences_real_len}

        val_data = {'sentences': fold_val_sentences,
                    'positions': fold_val_positions,
                    'y': fold_val_y,
                    'd_tags': fold_val_sentence_dependency_tags,
                    'real_len': fold_val_sentences_real_len}

        train_data_loader, val_data_loader, sentences_word2idx, pos_word2idx = generate_ds(train_data, val_data,
                                                                                           train_batch_size,
                                                                                           train_shuffle, max_seq_len)
        yield train_data_loader, val_data_loader, sentences_word2idx, pos_word2idx
