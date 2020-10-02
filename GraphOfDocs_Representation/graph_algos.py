class GraphAlgos:
    graph, database, = None, None

    def __init__(self, database, graph, start, relationship, end = None, property = None):
        # Initialize the class members.
        self.database = database
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
        self.database.execute(f'CALL gds.graph.create({graph_setup})', 'w')

    def pagerank(self, write_property, max_iterations = 20, damping_factor = 0.85):
        setup = (f'"{self.graph}", {{ '
            f'writeProperty: "{write_property}", '
            f'maxIterations: {max_iterations}, '
            f'dampingFactor: {damping_factor}}}'
        )
        self.database.execute(f'CALL gds.pageRank.write({setup})', 'w')


    # These methods enable the use of a with statement.
    def __enter__(self):
        return self

    # Automatic cleanup of the formerly used graph.
    def __exit__(self, exc_type, exc_value, traceback):
        self.database.execute(f'CALL gds.graph.drop("{self.graph}")', 'w')
