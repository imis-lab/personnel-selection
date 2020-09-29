class ModelEvaluator:
    """Model evaluation methods.
    """

    def evaluate_classifier(self, model, train_x, train_y, test_x, test_y):
        """Evaluate a given text classifier.

        :param model: the under evaluation classifier
        :param train_x: the train input
        :param train_y: the train encoded labels
        :param test_x: the test input
        :param test_y: the test encoded labels
        :return: a dictionary that contains (i) the model, (ii) the achieved accuracy score
        """
        model.fit(train_x, train_y)
        return {
            'model': model,
            'accuracy': model.score(test_x, test_y),
        }
