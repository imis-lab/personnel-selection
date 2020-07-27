from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class ClassifierGenerator:
    """Widely-used text classifiers.
    """

    def naive_bayes(self):
        """Generate a Multinomial naive Bayes text classifier.

        :return: a multinomial naive bayes classifier
        """
        return MultinomialNB()

    def knn(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """Generate a k nearest neighbors text classifier.

        :param n_neighbors: number of neighbors considered
        :param weights: how to treat the considered neighbors
        :return: a k-nn nearest neighbors classifier
        """
        return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def linear_svm(self):
        """Generate a linear support vector machine text classifier.

        :return: a linear support vector machine classifier
        """
        return LinearSVC()

    def decision_tree(self, max_depth: int = 5, random_state: int = 0):
        """Generate a decision tree text classifier.

        :param max_depth: the maxi depth of the tree
        :param random_state: the random state
        :return: a decision tree classifier
        """
        return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def logistic_regression(self, multi_class: str = 'multinomial', solver: str = 'lbfgs', random_state: int = 0):
        """Generate a logistic regression text classifier.

        :param multi_class: the type of classification
        :param solver: the utilized solver
        :param random_state: the random state
        :return: a logistic regression classifier
        """
        return LogisticRegression(random_state=random_state, solver=solver, multi_class=multi_class)

    def neural_network(self):
        """Not implemented yet.

        :return: a neural network classifier
        """
        pass
