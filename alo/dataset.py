#!/usr/bin/env python

import os
from abc import ABC, abstractmethod
from functools import reduce
from collections import Counter

import pandas as pd
import numpy as np
import nltk


DATA_PATH = 'data'


class DataSet(ABC):
    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def __getattr__(self, name, dataframe):
        pass



class WikiTalk(DataSet):

    COMMENTS_ENDING = '_annotated_comments.tsv'
    ANNOTATIONS_ENDING = '_annotations.tsv'

    def __init__(self, dataset_types=None):
        if dataset_types is None:
            self.dataset_types = ['attack', 'aggression', 'toxicity']

        self.datasets = {}
        self._load()

    def _load(self):
        for dataset_type in self.dataset_types:
            self.datasets[dataset_type] = self._load_dataset(dataset_type)

        # TODO - subsets
        dfs = [self.attack, self.aggression['aggression'], self.toxicity['toxicity']]
        self.datasets['merged'] = reduce(lambda l, r: l.join(r, how='inner'), dfs)

    def __getattr__(self, dataset_type):
        if dataset_type not in self.dataset_types and dataset_type != 'merged':
            raise AttributeError(f'{type(self)} object has no attribute {dataset_type}')
        
        return self.datasets[dataset_type]

    def _load_dataset(self, dataset):
        comments_path = os.path.join(DATA_PATH, dataset + self.COMMENTS_ENDING)
        annotations_path = os.path.join(DATA_PATH, dataset + self.ANNOTATIONS_ENDING)

        comments_df = pd.read_csv(comments_path, sep='\t', dtype={'rev_id': str})
        annotations_df = pd.read_csv(annotations_path, sep='\t', dtype={'rev_id': str})

        # fix toxicity df 'rev_id' column(not int)
        if dataset == 'toxicity':
            comments_df['rev_id'] = comments_df['rev_id'].apply(lambda x: x.replace('.0', ''))
            annotations_df['rev_id'] = annotations_df['rev_id'].apply(lambda x: x.replace('.0', ''))

        comments_df.set_index('rev_id', inplace=True)

        labels = annotations_df.groupby('rev_id')[dataset].mean() > 0.5
        comments_df[dataset] = labels

        return comments_df

    def data2matrix(self):
        sents = self.merged['comment'].values
        labels = self.merged[['attack', 'aggression', 'toxicity']].values

        return np.hstack((self.text2matrix(sents), labels.astype(int)))

    def text2matrix(self, sents, max_len=124):
        vocab = Counter([word for sent in sents for word in nltk.word_tokenize(sent)])
        idx2word = list(vocab.keys())
        word2idx = {w: i+1 for i, w in enumerate(idx2word)}  # 0 for padding

        # TODO - avoid two runs
        sent_vectors = [[word2idx[word] for word in nltk.word_tokenize(sent)] for sent in sents]
        sent_vectors = [s[:124] if len(s) > 124 else s + [0] * (124 - len(s)) for s in sent_vectors]

        return sent_vectors
