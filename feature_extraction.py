from typing import List

import gensim
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer


class RepresentationLearner:
    """Feature extraction and representation learning methods.
    """

    def bag_of_words(self, df: pd.DataFrame, column_name: str, stop_words=None, max_df: float = 1.0,
                     min_df: float = 1, ngram: tuple = (1, 1), binary: bool = False) -> CountVectorizer:
        """Convert a column of a given pandas dataFrame into bag-of-words vectors.

        This method wraps the CountVectorizer sklearn class.
        For further information check the documentation at: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html.

        :param df: the input pandas DataFrame
        :param column_name: the name of the column
        :param stop_words: the stopwords list, e.g. 'english'
        :param max_df: max document frequency
        :param min_df: min document frequency
        :param ngram: the considered n-grams
        :param binary: if true it creates binary bag-of-words vectors, i.e. it simply describes whether a word exists (one) or not (zero)
        :return: an already fitted bag of words vectorizer
        """
        cv = CountVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df, ngram_range=ngram, binary=binary)
        cv.fit(df[column_name])
        return cv

    def word2vec(self, tokenized_texts: List[list], size: int = 100, window: int = 5, min_count: int = 1,
                 skipgram: int = 1, workers: int = 4) -> gensim.models.word2vec.Word2Vec:
        """Build and train a word2vec model.

        This method wraps the Word2Vec gensim class.
        For further information check the documentation at: https://radimrehurek.com/gensim/models/word2vec.html.

        :param tokenized_texts: the input texts as lists of word tokens
        :param size: dimensionality of the word vectors
        :param window: the neighbors windows of each word
        :param min_count:  it ignores all words with total frequency lower than min_count
        :param skipgram: if 1 it runs by following the skip-gram architecture, if 0 it runs by following the CBOW architecture
        :param workers: the number of parallel workers
        :return: the trained word2vec model
        """
        model = Word2Vec(tokenized_texts, size=size, window=window, min_count=min_count, workers=workers, sg=skipgram)
        return model

    def doc2vec(self, tokenized_texts: List[list], size: int = 100, window: int = 5, min_count: int = 1,
                workers: int = 4) -> gensim.models.doc2vec.Doc2Vec:
        """Build and train a doc2vec model.

        This method wraps the Doc2Vec gensim class.
        For further information check the documentation at: https://radimrehurek.com/gensim/models/doc2vec.html.

        :param tokenized_texts: the input texts as lists of word tokens
        :param size: dimensionality of the word vectors
        :param window: the neighbors windows of each word
        :param min_count:  it ignores all words with total frequency lower than min_count
        :param workers: the number of parallel workers
        :return: the trained doc2vec model
        """
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_texts)]
        model = Doc2Vec(documents, vector_size=size, window=window, min_count=min_count, workers=workers)
        return model

    def glove(self):
        pass

    def fast_text(self):
        pass

    def node2vec(self):
        pass

    def random_projection(self):
        pass

    def graph_sage(self):
        pass
