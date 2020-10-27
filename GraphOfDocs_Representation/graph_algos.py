import time
import json
import traceback
import numpy as np

from statistics import mean
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class GraphAlgos:
    """
    Wrapper class which handle the graph algorithms 
    more efficiently, by abstracting repeating code.
    """
    database = None # Static variable shared across objects.

    def __init__(self, database, start, relationship, end = None, orientation = 'NATURAL'):
        # Initialize the static variable and class member.
        if GraphAlgos.database is None:
            GraphAlgos.database = database
        
        # Initialize the optional parameter.
        end = end if end is not None else start

        # Construct the projection of the anonymous graph.
        self.graph_projection = (
            f'{{nodeProjection: ["{start}", "{end}"], '
            'relationshipProjection: {'
            f'{relationship}: {{'
            f'type: "{relationship}", '
            f'orientation: "{orientation}"}}}}'
        )

    def pagerank(self, write_property, max_iterations = 20, damping_factor = 0.85):
        setup = (f'{self.graph_projection}, '
            f'writeProperty: "{write_property}", '
            f'maxIterations: {max_iterations}, '
            f'dampingFactor: {damping_factor}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.pageRank.write({setup})', 'w')

    def node2vec(self, write_property, embedding_size = 100, iterations = 1, walk_length = 80,
                 walks_per_node = 10, window_size = 10, walk_buffer_size = 1000):
        setup = (f'{self.graph_projection}, '
            f'writeProperty: "{write_property}", '
            f'embeddingSize: {embedding_size}, '
            f'iterations: {iterations}, '
            f'walkLength: {walk_length}, '
            f'walksPerNode: {walks_per_node}, '
            f'windowSize: {window_size}, '
            f'walkBufferSize: {walk_buffer_size}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.alpha.node2vec.write({setup})', 'w')

    def graphSage(self, write_property, embedding_size = 64, epochs = 1, max_iterations = 10,
                  aggregator = 'mean', activation_function = 'sigmoid', degree_as_property = True):
        setup = (f'{self.graph_projection}, '
            f'writeProperty: "{write_property}", '
            f'embeddingSize: {embedding_size}, '
            f'epochs: {epochs}, '
            f'maxIterations: {max_iterations}, '
            f'aggregator: "{aggregator}", '
            f'activationFunction: "{activation_function}", '
            f'degreeAsProperty: {degree_as_property}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.alpha.graphSage.write({setup})', 'w')

    def randomProjection(self, write_property, embedding_size = 100, max_iterations = 10,
                         sparsity = 3, normalize_l2 = False):
        setup = (f'{self.graph_projection}, '
            f'writeProperty: "{write_property}", '
            f'embeddingSize: {embedding_size}, '
            f'maxIterations: {max_iterations}, '
            f'sparsity: {sparsity}, '
            f'normalizeL2: {normalize_l2}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.alpha.randomProjection.write({setup})', 'w')

    def nodeSimilarity(self, write_property, write_relationship, cutoff = 0.5, top_k = 10):
        setup = (f'{self.graph_projection}, '
            f'writeProperty: "{write_property}", '
            f'writeRelationshipType: "{write_relationship}", '
            f'similarityCutoff: {cutoff}, '
            f'topK: {top_k}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.nodeSimilarity.write({setup})', 'w')

    @staticmethod
    def get_embeddings(write_property):
        query = (
            'MATCH (p:Person)-[:is_assigned_to]->(i:Issue) ' 
            f'WHERE EXISTS(i.{write_property}) '
            f'RETURN i.{write_property}, p.uname AS assignee'
        )
        return GraphAlgos.database.execute(query, 'r')

    @staticmethod
    def write_word_embeddings_to_csv(write_property, filepath):
        query = (
            f'MATCH (w:Word) WHERE EXISTS(w.{write_property}) '
            f'RETURN w.key, w.{write_property}'
        )
        with open(filepath, 'w', encoding = 'utf-8-sig', errors = 'ignore') as file:
            file.write('idx,word,embedding\n')
            for i, (word, embedding) in enumerate(GraphAlgos.database.execute(query, 'r')):
                file.write(f'{i},{word},"{embedding}"\n')

    @staticmethod
    def train_classifier(embeddings):
        # Unpack the embeddings and the assignees in X and Y separately.
        X, y = map(list, zip(*embeddings))

        # Transform y using the Label Encoder.
        y = preprocessing.LabelEncoder().fit_transform(y)

        # Split our dataset into train and test.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

        # Construct the classifier.
        LR = LogisticRegression(random_state = 0, multi_class = 'multinomial')

        # Train the classifier.
        LR.fit(X_train, y_train)

        # Predict the values.
        y_pred = LR.predict(X_test)

        # Print the classification report.
        print(classification_report(y_test, y_pred, labels = np.unique(y_pred)))


    # These methods enable the use of this class in a with statement.
    def __enter__(self):
        return self

    # Automatic cleanup of the created graph of this class.
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
