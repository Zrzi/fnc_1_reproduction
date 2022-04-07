import os

import numpy as np

from csv import DictReader

from config import data_base_path
from config import label_ref, label_ref_rev


def print_confusion_matrix(matrix):
    labels = ['unrelated', 'discuss', 'agree', 'disagree']
    lines = ['CONFUSION MATRIX:']
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *labels, 'overall')
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(matrix):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(labels[i], *row, sum(row)))
        lines.append("-"*line_len)
    print('\n'.join(lines))


if __name__ == '__main__':
    # 保存的结果文件地址
    predict_file_path = os.path.join(data_base_path, 'prediction.csv')
    # 真是标签地址
    label_file_path = os.path.join(data_base_path, 'competition_test_stances.csv')

    correct = 0
    total = 0
    score = 0.0
    confusion_matrix = np.zeros(shape=(4, 4), dtype=np.int)

    labels_dict = {}

    related = ['discuss', 'agree', 'disagree']

    max_score = 0.0
    null_score = 0.0

    with open(label_file_path, 'r', encoding='utf-8') as table:
        lines = DictReader(table)
        unrelated_count = 0
        total_count = 0
        for line in lines:
            line = dict(line)
            labels_dict[line['Headline'], line['Body ID']] = line['Stance']
            if line['Stance'] == 'unrelated':
                unrelated_count = unrelated_count + 1
            total_count = total_count + 1
        null_score = 0.25 * unrelated_count
        max_score = null_score + total_count - unrelated_count

    with open(predict_file_path, 'r', encoding='utf-8') as table:
        lines = DictReader(table)
        for line in lines:
            line = dict(line)
            predict = line['Stance']
            label = labels_dict[line['Headline'], line['Body ID']]

            if label == predict:
                correct += 1
                score += 0.25
                if label != 'unrelated':
                    score += 0.50

            if label in related and predict in related:
                score += 0.25

            total += 1
            confusion_matrix[label_ref[predict], label_ref[label]] += 1

    print_confusion_matrix(matrix=confusion_matrix)

    print()

    macro_f1 = 0
    for i in range(4):
        precision = confusion_matrix[i, i] / (confusion_matrix[i, :].sum())
        recall = confusion_matrix[i, i] / (confusion_matrix[:, i].sum())
        f1_score = 2 * precision * recall / (precision + recall)
        macro_f1 += f1_score
        print('{:^9} F1分数： {:.4f}'.format(label_ref_rev[i], f1_score))

    print()
    macro_f1 /= 4
    print('MACRO F1分数： {}'.format(macro_f1))

    print()

    print("总共{}个数据，正确{}个数据，正确率为{:.4f}".format(total, correct, 1.0 * correct / total))

    print()

    print('fnc-1 score: {}, fnc-1 relative score: {:.2f}'.format(score, score / max_score))
