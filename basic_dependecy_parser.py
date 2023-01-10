from torch import nn
import torch


class Mlp(nn.Module):
    def __init__(self, input_dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
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
    def __init__(self, device):
        super(DependencyParser, self).__init__()
        # self.word_embedding: word_embedding is implemented outside the model for ease of use
        self.device = device
        self.encoder = nn.LSTM(input_size=200, num_layers=2, bidirectional=True, hidden_size=256, batch_first=True)
        self.edge_scorer = Mlp(input_dim=256 * 2 * 2)
        self.loss_function = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, padded_sentence, padded_dependency_tree, real_seq_len):
        sentence = padded_sentence[:real_seq_len]
        dependency_tree = padded_dependency_tree[:real_seq_len]
        lstm_out, _ = self.encoder(sentence)

        new_word_embeddings = lstm_out[0]
        X1 = lstm_out.unsqueeze(0)
        Y1 = lstm_out.unsqueeze(1)
        X2 = X1.repeat(lstm_out.shape[0], 1, 1)
        Y2 = Y1.repeat(1, lstm_out.shape[0], 1)
        Z = torch.cat([Y2, X2], -1)
        lstm_out_combi = Z.view(-1, Z.shape[-1])
        score_mat_self_loop = self.edge_scorer(lstm_out_combi).view((lstm_out.shape[0], lstm_out.shape[0]))
        mask = torch.ones_like(score_mat_self_loop).fill_diagonal_(10000)

        scores_matrix = score_mat_self_loop - mask
        dependency_tree = dependency_tree.type(torch.LongTensor).to('cuda')
        loss = self.loss_function(self.log_softmax(scores_matrix)[1:], dependency_tree[1:])
        out_score_matrix = scores_matrix.T.fill_diagonal_(0)
        out_score_matrix[:, 0] = 0
        return loss, out_score_matrix
