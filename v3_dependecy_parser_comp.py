from torch import nn
import torch


class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)
        return output


class DependencyParser(nn.Module):
    def __init__(self, embedding_dim, sentences_word2idx, pos_word2idx, device):
        super(DependencyParser, self).__init__()
        self.embedding_dim = embedding_dim
        self.sentences_word2idx = sentences_word2idx
        self.pos_word2idx = pos_word2idx
        self.word_embedding = nn.Embedding(len(sentences_word2idx), self.embedding_dim)
        self.pos_embedding = nn.Embedding(len(pos_word2idx), self.embedding_dim)
        self.device = device
        self.encoder = nn.LSTM(input_size=400, num_layers=2, bidirectional=True, hidden_size=256, batch_first=True)
        self.edge_scorer = Mlp(input_dim=256 * 2 * 2, output_dim=1)
        self.loss_function = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.tags_classifier = Mlp(input_dim=256 * 2, output_dim=20)

    def forward(self, padded_sentence, padded_pos, real_seq_len, padded_d_tags=None, padded_dependency_tree=None):
        # rq = real_seq_len.item()
        sentence = padded_sentence[:real_seq_len]
        pos = padded_pos[:real_seq_len]

        sentence_embeddings = self.word_embedding(sentence)
        pos_embeddings = self.pos_embedding(pos)

        embeddings = torch.cat((sentence_embeddings, pos_embeddings), 1)

        lstm_out, _ = self.encoder(embeddings)

        predicted_d_tags = self.tags_classifier(lstm_out)

        X1 = lstm_out.unsqueeze(0)
        Y1 = lstm_out.unsqueeze(1)
        X2 = X1.repeat(lstm_out.shape[0], 1, 1)
        Y2 = Y1.repeat(1, lstm_out.shape[0], 1)
        Z = torch.cat([Y2, X2], -1)
        lstm_out_combi = Z.view(-1, Z.shape[-1])
        score_mat_self_loop = self.edge_scorer(lstm_out_combi).view((lstm_out.shape[0], lstm_out.shape[0]))
        mask = torch.ones_like(score_mat_self_loop).fill_diagonal_(10000)

        scores_matrix = score_mat_self_loop - mask
        out_score_matrix = scores_matrix.T.fill_diagonal_(0)
        out_score_matrix[:, 0] = 0
        if padded_dependency_tree is not None:
            d_tags = padded_d_tags[:real_seq_len]
            d_tags_loss = self.loss_function(self.log_softmax(predicted_d_tags), d_tags)

            dependency_tree = padded_dependency_tree[:real_seq_len]
            dependency_tree = dependency_tree.type(torch.LongTensor).to('cuda')
            loss = self.loss_function(self.log_softmax(scores_matrix)[1:], dependency_tree[1:]) + d_tags_loss
            return loss, out_score_matrix
        return out_score_matrix
