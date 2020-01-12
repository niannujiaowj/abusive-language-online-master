"""Word Embeddings"""

from abc import ABC, abstractmethod
import sys
import numpy as np


class WordEmbeddings(ABC):
    """
    Pre-trained word vectors
    """
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn
        self.word2embed = {}
        self.vocab_size, self.embed_dim = 0, 0

        self._load_vec()
        self.idx2word = list(self.word2embed)  # TODO no need to keep
        self.word2idx = {w: idx for idx, w in enumerate(self.idx2word)}

    @abstractmethod
    def _load_vec(self, dtype='float32'):
        """Factory Method"""
        pass

    def sent2vec(self, sent):
        """Convert sentence to vectors"""
        pass

    def add_word(self, word, word_vec=None):
        """Add one word vector pairt"""
        assert self.vocab_size != 0, 'Word vector not loaded'

        if word_vec is None:
            word_vec = self.rand_vector(self.embed_dim)

        if word not in self.word2embed:
            self.word2embed[word] = word_vec
            self.idx2word.append(word)
            self.word2idx[word] = self.vocab_size  # idx = new size - 1
            self.vocab_size += 1
        else:
            raise KeyError('Word embeddings exists')

        return self.vocab_size - 1  # index of added word

    def rand_vector(self, dim=None):
        """Generate a random vector"""
        if dim is None:
            dim = self.embed_dim
        return np.random.uniform(-0.25, 0.25, dim).astype(np.float32)


class Word2Vec(WordEmbeddings):
    """Google word2vec"""
    def _load_vec(self, dtype='float32'):
        """
        load compiled Word2Vec binary
        """
        with open(self.embedding_fn, "rb") as f:  # pylint: disable=invalid-name
            header = f.readline()
            self.vocab_size, self.embed_dim = [int(i) for i in header.split()]
            print('Vocabuarly size = {}, Vector dimensionality = {}'.format(
                self.vocab_size, self.embed_dim), flush=True, file=sys.stderr)

            block_size = np.dtype(dtype).itemsize * self.embed_dim
            for i in range(self.vocab_size):
                word = f.read(1)
                while not word.endswith(b' '):
                    word += f.read(1)
                word = word.strip()

                vector_bin = f.read(block_size)
                vector = np.fromstring(vector_bin, dtype=dtype)

                if word not in self.word2embed:
                    self.word2embed[word.decode()] = vector
                else:
                    raise KeyError('Duplicated key {}'.format(word))

                print(
                    'Loading.. {:.1f}% finished'.format(i * 100 / self.vocab_size),
                    end='\r', flush=True, file=sys.stderr)
        print('Binary resource successfully loaded', file=sys.stderr)


class Glove(WordEmbeddings):
    """Glove"""
    def _load_vec(self, dtype='float32'):
        pass


class FastText(WordEmbeddings):
    """FastText"""
    def _load_vec(self, dtype='float32'):
        pass
