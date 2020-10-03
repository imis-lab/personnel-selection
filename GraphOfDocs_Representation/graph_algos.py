import traceback

class GraphAlgos:
    """
    Wrapper class which handle the graph algorithms 
    more efficiently, by abstracting repeating code.
    """
    database = None # Static variable shared across objects.

    def __init__(self, database, graph, start, relationship, end = None, property = None):
        # Initialize the static variable and class member.
        if GraphAlgos.database is None:
            GraphAlgos.database = database
        self.graph = graph

        # Initialize the optional arguments.
        rel_properties = (None if property is None
                          else f'{{relationshipProperties: "{property}"}}')
        end = start if end is None else end

        # Setup the graph parameters.
        graph_setup = (
            f'"{self.graph}", ["{start}", "{end}"], "{relationship}"'
        )
        if rel_properties is not None:
            graph_setup = ', '.join((graph_setup, rel_properties))

        # Create the graph.
        GraphAlgos.database.execute(f'CALL gds.graph.create({graph_setup})', 'w')

    def pagerank(self, write_property, max_iterations = 20, damping_factor = 0.85):
        setup = (f'"{self.graph}", {{ '
            f'writeProperty: "{write_property}", '
            f'maxIterations: {max_iterations}, '
            f'dampingFactor: {damping_factor}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.pageRank.write({setup})', 'w')

    def node2vec(self, write_property, embedding_size = 10, iterations = 1, walk_length = 80,
                 walks_per_node = 10, window_size = 10, walk_buffer_size = 1000):
        setup = (f'"{self.graph}", {{ '
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
        setup = (f'"{self.graph}", {{ '
            f'writeProperty: "{write_property}", '
            f'embeddingSize: {embedding_size}, '
            f'epochs: {epochs}, '
            f'maxIterations: {max_iterations}, '
            f'aggregator: "{aggregator}", '
            f'activationFunction: "{activation_function}", '
            f'degreeAsProperty: {degree_as_property}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.alpha.graphSage.write({setup})', 'w')

    def randomProjection(self, write_property, embedding_size, max_iterations,
                         sparsity = 3, normalize_l2 = False):
        setup = (f'"{self.graph}", {{ '
            f'writeProperty: "{write_property}", '
            f'embeddingSize: {embedding_size}, '
            f'maxIterations: {max_iterations}, '
            f'sparsity: {sparsity}, '
            f'normalizeL2: {normalize_l2}}}'
        )
        GraphAlgos.database.execute(f'CALL gds.alpha.randomProjection.write({setup})', 'w')

    @staticmethod
    def get_embeddings(write_property):
        query = (
            'MATCH (p:Person)-[:is_assigned_to]->(i:Issue) '
            f'WHERE EXISTS(i.{write_property}) '
            f'RETURN i.key AS key, p.uname AS assignee, i.{write_property}'
        )
        return GraphAlgos.database.execute(query, 'r')

    # These methods enable the use of this class in a with statement.
    def __enter__(self):
        return self

    # Automatic cleanup of the created graph of this class.
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        GraphAlgos.database.execute(f'CALL gds.graph.drop("{self.graph}")', 'w')
