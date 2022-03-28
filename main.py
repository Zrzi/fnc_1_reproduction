import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os

from model import FNCModel

from data import FNCDataset
from data import pipeline_train, pipeline_test

from config import lim_unigram
from config import dropout
from config import batch_size
from config import learning_rate
from config import weight_decay
from config import epoch
from config import label_ref_rev
from config import data_base_path

from util import save_predictions


if __name__ == '__main__':
    # 1、获取数据
    train_dataset = FNCDataset(mode='train')
    test_dataset = FNCDataset(mode='test')

    # 2、处理数据
    bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer =\
        pipeline_train(train_dataset, test_dataset, lim_unigram=lim_unigram)
    pipeline_test(test_dataset, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    # 3、实例化数据加载器、模型、优化器等
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    model = FNCModel(2 * lim_unigram + 1, dropout)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 根据API，已经有L2正则化了

    # 4、训练模型
    model.train()
    for i in range(epoch):
        print('Epoch: {}/{}'.format(i + 1, epoch))
        for index, (text, label) in enumerate(train_data_loader):
            optimizer.zero_grad()
            output = model(text)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                print('第{}次，loss: {}'.format(index, loss.item()))

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