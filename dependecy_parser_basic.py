from torch import nn
from gensim import downloader
import numpy as np
from parser import parse_train_file
import torch
import itertools


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
    def __init__(self, word_embedding):
        super(DependencyParser, self).__init__()
        self.word_embedding = word_embedding
        self.encoder = nn.LSTM(input_size=200, num_layers=2, bidirectional=True, hidden_size=256)
        self.mlp_head = Mlp(input_dim=256 * 2 * 2)

    def embed_sentence(self, sen):
        representation = []
        for word in sen:
            word = word.lower()
            if word not in self.word_embedding.key_to_index:
                vec = np.zeros(200)
            else:
                vec = self.word_embedding[word]
            representation.append(vec)
        representation = np.asarray(representation)
        return representation

    def edge_scorer(self, edges, sentence_embedding):
        # table [i][j] == score of edge from vertex i to vertex j (row->column)
        # mlp head: [word_1_embeddings, word_2_embeddings] -> score for edge from word1 to word2
        score_matrix = torch.zeros(sentence_embedding.shape[0], sentence_embedding.shape[0])
        for vertex_1, vertex_2 in edges:
            word_1_embeddings = sentence_embedding[vertex_1]
            word_2_embeddings = sentence_embedding[vertex_2]
            edge_score = self.mlp_head(torch.concatenate([word_1_embeddings, word_2_embeddings]))
            score_matrix[vertex_1][vertex_2] = edge_score

        return score_matrix

    def forward(self, sentence, real_dependency_tree):
        basic_sentence_embedding = self.embed_sentence(sentence)
        basic_sentence_embedding = torch.Tensor(basic_sentence_embedding)
        new_word_embeddings, _ = self.encoder(basic_sentence_embedding)
        all_possible_edges_regular_sen = list(itertools.permutations(list(range(1, new_word_embeddings.shape[0])), 2))
        edges_from_root = [(0, i) for i in range(1, new_word_embeddings.shape[0])]
        all_possible_edges = edges_from_root + all_possible_edges_regular_sen
        # Get score for each possible edge in the parsing graph, construct score matrix
        score_matrix = self.edge_scorer(edges=all_possible_edges, sentence_embedding=new_word_embeddings)
        # Calculate the negative log likelihood loss described above
        loss = self.loss_function(real_dependency_tree, score_matrix)
        return loss, score_matrix

    def loss_function(self, real_dependency_tree, score_matrix):
        total_loss = 0
        for vertex_1, vertex_2 in real_dependency_tree[1:]: # first edge is a dummy edge to vertex ROOT
            normalizing_score_sum = sum([torch.exp(score_matrix[j][vertex_2]) if j != vertex_2 else 0 for j in
                                         range(len(real_dependency_tree))])
            softmax_score = torch.exp(score_matrix[vertex_1][vertex_2]) / normalizing_score_sum
            total_loss += -1 * torch.log(softmax_score)

        return total_loss / (len(real_dependency_tree)-1)


def main():
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    model = DependencyParser(word_embedding=glove)

    file_address = '/home/user/PycharmProjects/nlp_ex_3/data/train.labeled'
    sentences, sentence_tags = parse_train_file(file_address)
    loss, score_matrix = model(sentences[0], sentence_tags[0])
    print(loss)
    print(score_matrix.detach().numpy().shape)


if __name__ == '__main__':
    main()
