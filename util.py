import csv


def get_weights(train_dataset):
    """
    由于样本分布不平均，获取权重
    :param train_dataset: FNCDataset object, 训练集
    :return: list of weights
    """
    total = 0
    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    for text, label in train_dataset:
        counts[label] = counts[label] + 1
        total += 1

    # 避免出现计数为0的情况，平滑处理
    if counts[0] == 0:
        counts[0] += 1
        total += 1
    if counts[1] == 0:
        counts[1] += 1
        total += 1
    if counts[2] == 0:
        counts[2] += 1
        total += 1
    if counts[3] == 0:
        counts[3] += 1
        total += 1

    return [total / counts[label] for text, label in train_dataset]


def save_predictions(pred, file):

    """

    Save predictions to CSV file

    Args:
        pred: list, of dictionaries
        file: str, filename + extension

    """

    with open(file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Headline', 'Body ID', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for prediction in pred:
            writer.writerow(prediction)