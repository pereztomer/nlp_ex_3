import numpy as np


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


def main():
    file_address = '/home/user/PycharmProjects/nlp_ex_3/data/train.labeled'
    sentences, sentence_tags = parse_train_file(file_address)
    print('hi')


if __name__ == '__main__':
    main()
