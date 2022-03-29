import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import torch.nn.functional as F

import os

from model import FNCModel

from data import FNCDataset, pipeline_train, pipeline_test

from config import lim_unigram
from config import dropout
from config import batch_size
from config import learning_rate
from config import weight_decay
from config import epoch
from config import label_ref_rev
from config import data_base_path

from util import save_predictions, get_weights

if __name__ == '__main__':
    # 1、获取数据
    train_all_dataset = FNCDataset(mode='train')
    test_dataset = FNCDataset(mode='test')

    # 2、处理数据
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(train_all_dataset, test_dataset, lim_unigram=lim_unigram)
    pipeline_test(test_dataset, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    # 划分数据集，按照9:1划分
    train_len = int(len(train_all_dataset) * 9 / 10)
    validation_len = len(train_all_dataset) - train_len

    # 实例化dataloader
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # 3、，实例化数据加载器、模型、优化器等
    model = FNCModel(2 * lim_unigram + 1, dropout)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 根据API，已经有L2正则化了

    # 4、训练模型
    for i in range(epoch):

        train_dataset, validation_dataset = \
            random_split(dataset=train_all_dataset, lengths=[train_len, validation_len])
        weights = get_weights(train_dataset)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        print('Epoch: {}/{}'.format(i + 1, epoch))
        for index, (text, label) in enumerate(train_data_loader, start=1):
            optimizer.zero_grad()
            output = model(text)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print('第{}次，loss: {}'.format(index, loss.item()))

        # 在验证集上测试，评估超参数
        model.eval()
        total_loss = 0
        correct = 0
        for index, (text, label) in enumerate(validation_data_loader, start=1):
            with torch.no_grad():
                output = model(text)
                loss = F.nll_loss(output, label)
                total_loss += loss.item()
                predict = output.argmax(dim=1)
                correct += predict.eq(label).sum().item()
        print('验证集结果，total_loss: {}，正确{}条数据，准确率是{}'.format(total_loss, correct, 1.0 * correct / validation_len))

    # 5、test
    model.eval()
    results = []
    for text, label, prev_text, prev_label in test_data_loader:
        with torch.no_grad():
            output = model(text)
            predict = output.max(dim=-1)[-1].item()
            predict = label_ref_rev[predict]
            result = {
                'Headline': prev_text[0][0],
                'Body ID': prev_text[2].item(),
                'Stance': predict
            }
            results.append(result)

    # 6、保存结果
    prediction_path = os.path.join(data_base_path, 'prediction.csv')
    save_predictions(results, prediction_path)
