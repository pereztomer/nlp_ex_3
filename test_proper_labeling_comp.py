import numpy as np


def parse_train_file(file_address):
    with open(file_address, encoding='utf-8') as f1:
        with open('val_untaged.txt', 'w') as f2:
            for row in f1:
                if row != '\n':
                    token = row.split('\t')[1]
                    token_pos = row.split('\t')[3]
                    token_head = row.split('\t')[6]
                    dependency_label = row.split('\t')[7]

                    row_lst = row.split('\t')
                    row_lst[6] = '_'
                    row_lst[7] = '_'
                    row_string = ''
                    for idx, x in enumerate(row_lst):
                        if idx == len(row_lst) - 1:
                            row_string = row_string + x
                        else:
                            row_string = row_string + str(x) + '\t'

                    f2.write(row_string)
                    row_string = ''
                else:
                    f2.write('\n')


def extract_labels(address):
    y = []
    row_counter = 0
    with open(address, 'r') as f:
        for row in f:
            row_counter += 1
            if row != '\n':
                token = row.split('\t')[1]
                token_pos = row.split('\t')[3]
                token_head = row.split('\t')[6]
                y.append(token_head)
                dependency_label = row.split('\t')[7]

    print(row_counter)

    return y


def calc_uas():
    val_address = '/home/user/PycharmProjects/nlp_ex_3/data/test.labeled'
    true_labels = extract_labels(val_address)

    val_predicted = '/home/user/PycharmProjects/nlp_ex_3/competetion_v3/val_untaged_taged.txt'
    # val_predicted = '/home/user/PycharmProjects/nlp_ex_3/val_untaged.txt'
    predicted_labels = extract_labels(val_predicted)

    uas_loss_lst = (np.array(true_labels) == np.array(predicted_labels))
    # uas_loss_lst += list((mst[1:] == y[1:real_seq_len].detach().cpu().numpy()))
    # total_tokens_num += len(mst[1:])
    # uas_loss_lst.append(uas_loss)

    uas = sum(uas_loss_lst) / len(uas_loss_lst)
    print(f'uas: {uas}')


def main():
    val_address = '/home/user/PycharmProjects/nlp_ex_3/data/test.labeled'
    parse_train_file(val_address)


if __name__ == '__main__':
    calc_uas()
