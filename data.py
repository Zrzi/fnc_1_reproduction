import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset

import os
import csv

from config import stop_words
from config import data_base_path
from config import label_ref


class FNCDataset(Dataset):
    def __init__(self, mode='train'):
        super(FNCDataset, self).__init__()
        self.mode = mode
        if mode == 'train':
            data_stances_path = os.path.join(data_base_path, 'train_stances.csv')
            data_bodies_path = os.path.join(data_base_path, 'train_bodies.csv')
        else:
            data_stances_path = os.path.join(data_base_path, 'test_stances_unlabeled.csv')
            data_bodies_path = os.path.join(data_base_path, 'test_bodies.csv')

        stances = self.read(data_stances_path)
        bodies = self.read(data_bodies_path)

        body = {}
        self.pairs = []

        for aBody in bodies:
            body[int(aBody['Body ID'])] = aBody['articleBody']

        for stance in stances:
            stance = dict(stance)
            pair = {}
            if self.mode == 'train':
                pair['text'] = (stance['Headline'], body[int(stance['Body ID'])])
            else:
                pair['text'] = (stance['Headline'], body[int(stance['Body ID'])], int(stance['Body ID']))
            pair['label'] = stance['Stance'] if self.mode == 'train' else 'to be predicted'
            self.pairs.append(pair)

        self.previous_pairs = self.pairs

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.pairs[index]['text'], self.pairs[index]['label']
        else:
            return self.pairs[index]['text'], self.pairs[index]['label'],\
                   self.previous_pairs[index]['text'], self.previous_pairs[index]['label']

    def __len__(self):
        return len(self.pairs)

    def read(self, filename):
        rows = []

        with open(filename, 'r', encoding='utf-8') as table:
            r = csv.DictReader(table)
            for line in r:
                rows.append(line)

        return rows


# Define relevant functions
def pipeline_train(train, test, lim_unigram):
    """

    Process train set, create relevant vectorizers

    Args:
        train: FNCDataset object, train set
        test: FNCDataset object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    """

    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    id_ref = {}
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    head_tfidf_track = {}
    body_tfidf_track = {}

    # Identify unique heads and bodies
    for text, label in train:
        headline = text[0]
        body = text[1]
        if headline not in heads_track:
            heads.append(headline)
            heads_track[headline] = 1
        if body not in bodies_track:
            bodies.append(body)
            bodies_track[body] = 1

    for text, label, pre_text, pre_label in test:
        headline = text[0]
        body = text[1]
        if headline not in test_heads_track:
            test_heads.append(headline)
            test_heads_track[headline] = 1
        if body not in test_bodies_track:
            test_bodies.append(body)
            test_bodies_track[body] = 1

    # Create reference dictionary
    for i, elem in enumerate(heads + bodies):
        id_ref[elem] = i

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words). \
        fit(heads + bodies + test_heads + test_bodies)  # Train and test sets

    new_pairs = []

    # Process train set
    for text, label in train:
        headline = text[0]
        body = text[1]
        head_tf = tfreq[id_ref[headline]].reshape(1, -1)
        body_tf = tfreq[id_ref[body]].reshape(1, -1)
        if headline not in head_tfidf_track:
            head_tfidf = tfidf_vectorizer.transform([headline]).toarray()
            head_tfidf_track[headline] = head_tfidf
        else:
            head_tfidf = head_tfidf_track[headline]
        if body not in body_tfidf_track:
            body_tfidf = tfidf_vectorizer.transform([body]).toarray()
            body_tfidf_track[body] = body_tfidf
        else:
            body_tfidf = body_tfidf_track[body]
        if (headline, body) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(headline, body)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(headline, body)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        pair = {}
        pair['text'] = torch.from_numpy(feat_vec)
        pair['label'] = label_ref[label]
        new_pairs.append(pair)

    train.pairs = new_pairs

    return bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
    """

    Process test set

    Args:
        test: FNCDataset object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:

    """

    # Initialise
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    new_pairs = []

    # Process test set
    for text, label, pre_text, pre_label in test:
        headline = text[0]
        body = text[1]
        if headline not in heads_track:
            head_bow = bow_vectorizer.transform([headline]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([headline]).toarray().reshape(1, -1)
            heads_track[headline] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[headline][0]
            head_tfidf = heads_track[headline][1]
        if body not in bodies_track:
            body_bow = bow_vectorizer.transform([body]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)
            bodies_track[body] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body][0]
            body_tfidf = bodies_track[body][1]
        if (headline, body) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(headline, body)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(headline, body)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        pair = {}
        pair['text'] = torch.from_numpy(feat_vec)
        pair['label'] = 'unknown'
        new_pairs.append(pair)

    test.previous_pairs = test.pairs
    test.pairs = new_pairs