import re
import string
from typing import List, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

english_stopwords = stopwords.words('english')


def get_top_n_most_frequent_labels(df: pd.DataFrame, column_name: str, top_n: int) -> list:
    """Get the top-n most frequent labels.

    :param df: the input pandas DataFrame
    :param column_name: the name of the column
    :param top_n: the number of the top-n most frequent labels included
    :return: a list with the top-n most frequent labels
    """
    return df[column_name].value_counts()[:top_n].index.to_list()


def fit_label_encoder(df: pd.DataFrame, column_name: str) -> LabelEncoder:
    """Create a label encoder.

    This function maps alphanumeric labels to unique numbers.
    It fits the label encoder into the given dataFrame and returns it.

    :param df: the input pandas DataFrame
    :param column_name: the name of the column that contains the labels
    :return: a fitted label encoder
    """
    le = LabelEncoder()
    le.fit(df[column_name])
    return le


class TextPreprocessor:
    """Text preprocessing methods.
    """

    def preprocess_df(self, df: pd.DataFrame, column_name: str, lowercase: bool = True, remove_numerics: bool = True,
                      remove_stopwords: bool = True,
                      remove_punctuation: bool = True, stemming: bool = False, stringifized=False,
                      stopwords=english_stopwords) -> List[Union[list, str]]:
        """Preprocess a DataFrame column of texts.

        :param df: the input pandas DataFrame
        :param column_name: the name of the column
        :param lowercase: if true it converts all the uppercase letter to lowercase ones
        :param remove_numerics: if true it removes of all the numbers
        :param remove_stopwords: if true it removes all the stopwords contained in stopwords variable
        :param remove_punctuation: if true it removes all the punctuation marks
        :param stemming: if true it stems the text
        :param stringifized: if true it converts the list of tokens of each row to a text again
        :param stopwords: the list of the stopwords
        :return: a list of lists of word tokens
        """
        texts = df[column_name]
        return [self.preprocess(text, lowercase=lowercase, remove_numerics=remove_numerics,
                                remove_stopwords=remove_stopwords, remove_punctuation=remove_punctuation,
                                stemming=stemming, stringifized=stringifized, stopwords=stopwords) for text in texts]

    def preprocess(self, text: str, lowercase: bool = True, remove_numerics: bool = True, remove_stopwords: bool = True,
                   remove_punctuation: bool = True, stemming: bool = False, stringifized=False,
                   stopwords=english_stopwords) -> Union[list, str]:
        """Preprocess a given text.

        It removes (i) unnecessary whitespace and punctuation characters, (ii) stopwords, (iii) numbers, and
        performs stemming.

        :param text: the input text
        :param lowercase: if true it converts all the uppercase letter to lowercase ones
        :param remove_numerics: if true it removes of all the numbers
        :param remove_stopwords: if true it removes all the stopwords contained in stopwords variable
        :param remove_punctuation: if true it removes all the punctuation marks
        :param stemming: if true it stems the text
        :param stringifized: if true it converts the list of tokens to a text again
        :param stopwords: the list of the stopwords
        :return: a list of word tokens or a preprocessed str
        """
        if lowercase:
            text = self.lowercase(text)
        word_tokens = self.tokenize(text)
        if remove_numerics:
            word_tokens = [word for word in word_tokens if not word.isnumeric()]
        if remove_stopwords:
            word_tokens = [word for word in word_tokens if not word in stopwords]
        if remove_punctuation:
            word_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in word_tokens]
            word_tokens = [word for word in word_tokens if word != '']
        if remove_numerics:
            word_tokens = [re.sub('\d', '', word) for word in word_tokens]
        if stemming:
            word_tokens = self.stem(word_tokens)

        if stringifized:
            return ' '.join(word_tokens)
        return word_tokens

    def tokenize_df(self, df: pd.DataFrame, column_name: str) -> List[list]:
        return [self.tokenize(text) for text in df[column_name]]

    def tokenize(self, text: str) -> list:
        """Return the given text as word tokens.

        The nltk.word_tokenize is used to tokenize the text.

        :param text: the input text
        :return: a list of word tokens
        """
        return nltk.word_tokenize(text)

    def clean_text(self, text: str) -> str:
        return text

    def lowercase(self, text: str) -> str:
        """Convert all the letters of a text to lowercase.

        :param text: the input text
        :return: a text with all the letters in lowercase
        """
        return text.lower()

    def stem(self, words: list) -> list:
        """Stem word tokens.

        nltk.stem.PorterStemmer is used as a stemmer.

        :param words: the given word tokens
        :return: the given words stemmed
        """
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        return words
