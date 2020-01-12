#!/usr/bin/env python

import os
import numpy as np
import sys
from collections import Counter
import nltk
import pandas as pd
import matplotlib.pyplot as plt

SENT_LEN = 124

sys.path.insert(0, '/home/izoom/xyue/del')
import zz

wiki = zz.WikiTalk()
data = wiki.load()

print(data.shape)
print(data[data['label'] == 0].shape)
print(data[data['label'] == 1].shape)
print(data[data['label'] == 2].shape)
print(data[data['label'] == 3].shape)

t0 = data[data['label'] == 0]
t1 = data[data['label'] == 1]
t2 = data[data['label'] == 2]
t3 = data[data['label'] == 3]
t = [t0.iloc[:15362], t1, t2, t3]

data = pd.concat(t)
print(data.shape)
print(data[data['label'] == 0].shape)
print(data[data['label'] == 1].shape)
print(data[data['label'] == 2].shape)
print(data[data['label'] == 3].shape)

data['comment'] = data['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
data['comment'] = data['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

texts, labels = data['comment'].values, data['label'].values

fdist = nltk.FreqDist([word for sent in texts for word in nltk.word_tokenize(sent)])
idx2word = list(fdist.keys())
word2idx = {w: i+1 for i, w in enumerate(idx2word)}  # 0 for padding

sents = [[word2idx[word] for word in nltk.word_tokenize(sent)] for sent in texts]
sents = [s[:124] if len(s) > 124 else s + [0] * (124 - len(s)) for s in sents]

X = np.array(sents)
y = np.array(labels)

print(X.shape, y.shape)
#data = np.hstack((X, y))
#np.save('data', data)


#ldist = Counter([len(s) for s in sents])
#df = pd.DataFrame(list(ldist.items()), columns=['len', 'num'])
#df.to_pickle('df')
#df = df.sort_values('len')
#df['cumsum'] = df['num'].cumsum()
#df['percent'] = df['cumsum'] / df['cumsum'].iloc[-1] * 100
#print(df.head())

#for k,v in ldist.items():
#    print(k, v)

#plt.bar(ldist.keys(), ldist.values(), 1)
#plt.xticks(indexes + width * 0.5, labels)
#plt.show()


