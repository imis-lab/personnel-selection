import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


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

    def clean_text(self, text: str) -> str:
        """"Remove unnecessary characters from a given text.

        It removes unnecessary whitespace and punctuation characters of a text.
        Note: Not all of the punctuation characters are removed.

        :param text: the given text
        :returns: a text without the unnecessary whitespace and punctuation characters
        """
        text = text.replace('\r', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('. ', ' ')
        text = text.replace('!', ' ')
        text = text.replace('"', ' ')
        text = text.replace("'", ' ')
        text = text.replace("$", ' ')
        text = text.replace("#", ' ')
        text = text.replace("/", ' ')
        text = text.replace(";", ' ')
        text = text.replace("`", ' ')
        text = text.replace('  ', ' ')
        return text

    def stem_text(self, text: str) -> str:
        """Stem word tokens of a given text.

        nltk.stem.PorterStemmer is used as a stemmer.

        :param text: the given text
        :return: the given text with stemmed words
        """
        stemmer = PorterStemmer()
        words = nltk.word_tokenize(text)
        text = ' '.join([stemmer.stem(word) for word in words])
        return text


class FeatureExtractorGenerator:
    def bag_of_words_vectorizer(self, df: pd.DataFrame, column_name: str, stop_words=None, max_df: float = 1.0,
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
