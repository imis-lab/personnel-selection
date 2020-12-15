import time
import traceback
import networkx as nx
import pandas as pd
import numpy as np
import os
import random

from neo4j.types.graph import Node, Relationship
from node2vec import Node2Vec

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph import globalvar

class inMemoryGraph:
    """
    Wrapper class which converts a subgraph from the graph
    database into an in-memory graph using NetworkX.
    Node IDs correspond to the neo4j graph.
    Nodes have fields 'labels' (frozenset) and 'properties' (dicts).
    Edges have fields 'type_' (string) denoting the type of relation, and 'properties' (dict).
    """
    database = None # Static variable shared across objects.

    def __init__(self, database):
        # Initialize the static variable and class member.
        if inMemoryGraph.database is None:
            inMemoryGraph.database = database
        
        # Return the GraphOfDocs graph.
        query = 'MATCH (w:Word)-[r:connects]->(w2:Word) RETURN *'
        start = time.perf_counter()
        data = database.execute(query, 'g')
        end = time.perf_counter()
        print(f'Retrieving data {end-start} sec')

        start = time.perf_counter()
        # Construct the networkX Graph.
        self.G = nx.MultiDiGraph()

        for d in data:
            for entry in d.values():
                # Parse node.
                if isinstance(entry, Node):
                    self.__add_node(entry)

                # Parse link.
                elif isinstance(entry, Relationship):
                    self.__add_edge(entry)
                else:
                    raise TypeError("Unrecognized object")
        end = time.perf_counter()
        print(f'Constructing the in-memory graph {end-start} sec')

    # Private method that adds a node.
    def __add_node(self, node):
        # Adds node id it hasn't already been added
        u = node.id
        if self.G.has_node(u):
            return
        self.G.add_node(u, labels=node._labels, properties=dict(node))

    # Private method that adds an edge.
    def __add_edge(self, relation):
        # Adds edge if it hasn't already been added.
        # Make sure the nodes at both ends are created.
        for node in (relation.start_node, relation.end_node):
            self.__add_node(node)
        # Check if edge already exists.
        u = relation.start_node.id
        v = relation.end_node.id
        eid = relation.id
        if self.G.has_edge(u, v, key = eid):
            return
        # If not, create it.
        self.G.add_edge(u, v, key = eid, type_ = relation.type, properties = dict(relation))

    def node2vec(self, filepath, dimensions = 10, walk_length = 10, num_walks = 10, workers = 1):
        node2vec = Node2Vec(self.G, dimensions, walk_length, num_walks, workers)  # Use temp_folder for big graphs
        model = node2vec.fit(window = 4, min_count = 1, batch_words = 4)
        print(model.wv.most_similar('python'))
        model.wv.save_word2vec_format(filepath)

    def graphsage(self, walk_length = 4, number_of_walks = 1, epochs = 1, workers = 4):
        # Construct the node features based on the node id.
        for node_id, node_data in self.G.nodes(data = True):
            node_data["feature"] = [node_id, len(str(node_id))]

        # Construct the Stellar Graph from the NetworkX graph.
        G = StellarGraph.from_networkx(self.G, node_features = "feature")

        # Create the unsupervised samples.
        unsupervised_samples = UnsupervisedSampler(
            G, nodes = list(G.nodes()), 
            length = walk_length, number_of_walks = number_of_walks
        )
        # Create the node pair generator.
        generator = GraphSAGELinkGenerator(G, 50, [10, 5])
        train_gen = generator.flow(unsupervised_samples)

        # Create the graphsage encoder.
        graphsage = GraphSAGE(
            layer_sizes = [50, 50], 
            generator = generator, bias = True,
            dropout = 0.0, normalize = "l2"
        )

        # Build the model and expose input and output sockets of graphsage, for node pair inputs:
        x_inp, x_out = graphsage.in_out_tensors()

        prediction = link_classification(
            output_dim = 1, output_act = "sigmoid", edge_embedding_method = "ip"
        )(x_out)

        model = keras.Model(inputs = x_inp, outputs = prediction)

        model.compile(
            optimizer = keras.optimizers.Adam(lr = 1e-3),
            loss = keras.losses.binary_crossentropy,
            metrics = [keras.metrics.binary_accuracy],
        )

        history = model.fit(
            train_gen,
            epochs = epochs,
            verbose = 1,
            use_multiprocessing = False,
            workers = workers,
            shuffle = True,
        )

        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(inputs = x_inp_src, outputs = x_out_src)

        node_ids = node_subjects.index
        node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
        node_embeddings = embedding_model.predict(node_gen, workers = workers, verbose = 1)

    # These methods enable the use of this class in a with statement.
    def __enter__(self):
        return self

    # Automatic cleanup of the created graph of this class.
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)