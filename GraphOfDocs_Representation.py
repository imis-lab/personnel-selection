import sys
import time
from neo4j import ServiceUnavailable
from GraphOfDocs_Representation.neo4j_wrapper import Neo4jDatabase
from GraphOfDocs_Representation.graph_algos import GraphAlgos
from GraphOfDocs_Representation.in_memory_graph import inMemoryGraph
from GraphOfDocs_Representation.create import *

def graphofdocs(create, initialize, dirpath):
    # Open the database.
    try:
        database = Neo4jDatabase('bolt://localhost:7687', 'neo4j', '123')
        # Neo4j server is unavailable.
        # This client app cannot open a connection.
    except ServiceUnavailable as error:
        print('\t* Neo4j database is unavailable.')
        print('\t* Please check the database connection before running this app.')
        input('\t* Press any key to exit the app...')
        sys.exit(1)

    if create:
        # Delete nodes from previous iterations.
        database.execute('MATCH (n) DETACH DELETE n', 'w')

        # Create uniqueness constraint on key to avoid duplicate word nodes.
        create_unique_constraints(database)

        # Create papers and their citations, authors and their affiliations,
        # and the graph of words for each abstract, 
        # which is a subgraph of the total graph of docs.
        start = time.perf_counter()
        create_issues_from_json(database, dirpath)
        end = time.perf_counter()
        print(f'Create papers {end-start} sec')
        
    if initialize:
        # Run initialization functions.
        with inMemoryGraph(database) as graph:
            print('Here')

    database.close()
    return

if __name__ == '__main__': graphofdocs(False, True, r'C:\Users\USER\Desktop\issues.json')