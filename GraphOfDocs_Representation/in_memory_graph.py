import time
import networkx as nx
from neo4j.types.graph import Node, Relationship

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
        G = nx.MultiDiGraph()

        def add_node(node):
            # Adds node id it hasn't already been added
            u = node.id
            if G.has_node(u):
                return
            G.add_node(u, labels=node._labels, properties=dict(node))

        def add_edge(relation):
            # Adds edge if it hasn't already been added.
            # Make sure the nodes at both ends are created.
            for node in (relation.start_node, relation.end_node):
                add_node(node)
            # Check if edge already exists.
            u = relation.start_node.id
            v = relation.end_node.id
            eid = relation.id
            if G.has_edge(u, v, key=eid):
                return
            # If not, create it.
            G.add_edge(u, v, key=eid, type_=relation.type, properties=dict(relation))

        for d in data:
            for entry in d.values():
                # Parse node.
                if isinstance(entry, Node):
                    add_node(entry)

                # Parse link.
                elif isinstance(entry, Relationship):
                    add_edge(entry)
                else:
                    raise TypeError("Unrecognized object")
        end = time.perf_counter()
        print(f'Constructing the in-memory graph {end-start} sec')

    # These methods enable the use of this class in a with statement.
    def __enter__(self):
        return self

    # Automatic cleanup of the created graph of this class.
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
