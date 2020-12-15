import sys
import time
import pickle
from neo4j import ServiceUnavailable
from GraphOfDocs_Representation.neo4j_wrapper import Neo4jDatabase
from GraphOfDocs_Representation.graph_algos import GraphAlgos
from GraphOfDocs_Representation.create import *
from GraphOfDocs_Representation.select import *

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

        # Create issues from json using the GraphOfDocs model.
        start = time.perf_counter()
        create_issues_from_json(database, dirpath)
        end = time.perf_counter()
        print(f'Create papers {end-start} sec')

        # Create the similarity graph of topN = 10 similar words using emb. dim. = 300
        start = time.perf_counter()
        create_word2vec_similarity_graph(database, dirpath, 'jira_issues_300.model', 300)
        end = time.perf_counter()
        print(f'Created similarity graph in {end-start} sec')

    if initialize: # Run initialization functions.
        # Calculate the word embeddings for the word similarity graph and write them to csv files.
        with GraphAlgos(database, 'Word', 'similar_w2v', 'Word', rel_weight = 'score') as graph:
            for dim in [100, 200, 300]:
                # Generate the embeddings in the database.
                graph.graphSage(f'gs_{dim}', embedding_dim = dim)
                graph.node2vec(f'n2v_{dim}', embedding_dim = dim)
                graph.fastRP(f'fastrp_{dim}', embedding_dim = dim)

                graph.graphSage(f'gs_weighted_{dim}', embedding_dim = dim, rel_weight = 'score')
                graph.fastRP(f'fastrp_weighted_{dim}', embedding_dim = dim, rel_weight = 'score')

                # Export the embeddings in csv.
                graph.write_word_embeddings_to_csv(f'gs_{dim}', f'gs_{dim}.csv')
                graph.write_word_embeddings_to_csv(f'n2v_{dim}', f'n2v_{dim}.csv')
                graph.write_word_embeddings_to_csv(f'fastrp_{dim}', f'fastrp_{dim}.csv')

                graph.write_word_embeddings_to_csv(f'gs_weighted_{dim}', f'gs_weighted_{dim}.csv')
                graph.write_word_embeddings_to_csv(f'fastrp_weighted_{dim}', f'fastrp_weighted_{dim}.csv')

        # Construct the Issue similarity graph and calculate its communities.
        with GraphAlgos(database, 'Issue', 'includes', 'Word') as graph:
            graph.nodeSimilarity(write_property = 'score', write_relationship = 'is_similar', cutoff = 0.25, top_k = 1)
        with GraphAlgos(database, 'Issue', 'is_similar', 'Issue') as graph:
            graph.louvain(write_property = 'community')

        # Save the n top terms of a community of similar document to a pickle.
        for n in [5, 10, 15, 20, 25, 50, 100, 250, 500, 1000]:
            x = get_communities_tags(database, top_terms = n)
            tags = sum(x.values(), [])
            with open(f'features_{n}.pkl', 'wb') as f:
                pickle.dump(tags, f)
    database.close()
    return

if __name__ == '__main__': graphofdocs(False, True, 'issues_55k.json')