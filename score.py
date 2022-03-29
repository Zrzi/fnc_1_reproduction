import os

import numpy as np

from csv import DictReader

from config import data_base_path
from config import label_ref, label_ref_rev

if __name__ == '__main__':
    # 保存的结果文件地址
    predict_file_path = os.path.join(data_base_path, 'prediction.csv')
    # 真是标签地址
    label_file_path = os.path.join(data_base_path, 'competition_test_stances.csv')

    correct = 0
    total = 0
    confusion_matrix = np.zeros(shape=(4, 4), dtype=np.int)

    labels_dict = {}

    with open(label_file_path, 'r', encoding='utf-8') as table:
        lines = DictReader(table)
        for line in lines:
            line = dict(line)
            labels_dict[line['Headline'], line['Body ID']] = line['Stance']

    with open(predict_file_path, 'r', encoding='utf-8') as table:
        lines = DictReader(table)
        for line in lines:
            line = dict(line)
            predict = line['Stance']
            label = labels_dict[line['Headline'], line['Body ID']]
            if label == predict:
                correct += 1
            total += 1
            confusion_matrix[label_ref[predict], label_ref[label]] += 1

    # with open(label_file_path, 'r', encoding='utf-8') as f1, open(predict_file_path, 'r', encoding='utf-8') as f2:
    #     labels = DictReader(f1)
    #     predicts = DictReader(f2)
    #     for label, predict in zip(labels, predicts):
    #         label = dict(label)['Stance']
    #         predict = dict(predict)['Stance']
    #         if label == predict:
    #             correct += 1
    #         total += 1
    #         confusion_matrix[label_ref[label], label_ref[predict]] += 1

    print(confusion_matrix)

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
