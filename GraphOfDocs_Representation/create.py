"""
This script contains functions that 
create data in the Neo4j database.
"""
import json
import platform
from GraphOfDocs_Representation.utils import (
    clear_screen, generate_words
)

# Initialize an empty set of edges.
edges = {}
# Initialize an empty list of unique terms.
# We are using a list to preserver order of appearance.
nodes = []

def create_graph_of_words(words, database, filename, relationship, window_size = 4):
    """
    Function that creates a Graph of Words that contains all nodes from each document for easy comparison,
    inside the neo4j database, using the appropriate cypher queries.
    """

    # Files that have word length < window size, are skipped.
    # Window size ranges from 2 to 6.
    length = len(words)
    if (length < window_size):
        # Early exit, we return the skipped filename
        return filename

    # We are using a global set of edges to avoid creating duplicate edges between different graph of words.
    # Basically the co-occurences will be merged.
    global edges

    # We are using a global set of edges to avoid creating duplicate nodes between different graph of words.
    # A list is being used to respect the order of appearance.
    global nodes

    # We are getting the unique terms for the current graph of words.
    terms = []
    creation_list = []
    for word in words:
        if word not in terms: 
            terms.append(word)
    # Remove end-of-sentence token, so it doesn't get created.
    if 'e5c' in terms:
        terms.remove('e5c')
    # If the word doesn't exist as a node, then add it to the creation list.
    for word in terms:
        if word not in nodes:
            creation_list.append(word)
            # Append word to the global node graph, to avoid duplicate creation.
            nodes.append(word)

    # Create all unique nodes, from the creation list.
    database.execute(f'UNWIND {creation_list} as key '
                      'CREATE (word:Word {key: key})', 'w')

    # Create unique connections between existing nodes of the graph.
    for i, current in enumerate(words):
        # If there are leftover items smaller than the window size, reduce it.
        if i + window_size > length:
            window_size = window_size - 1
        # If the current word is the end of sentence string,
        # we need to skip it, in order to go to the words of the next sentence,
        # without connecting words of different sentences, in the database.
        if current == 'e5c':
            continue
        # Connect the current element with the next elements of the window size.
        for j in range(1, window_size):
            next = words[i + j]
            # Reached the end of sentence string.
            # We can't connect words of different sentences,
            # therefore we need to pick a new current word,
            # by going back out to the outer loop.
            if next == 'e5c':
                break
            edge = (current, next)
            if edge in edges:
                # If the edge, exists just update its weight.
                edges[edge] = edges[edge] + 1
                query = (f'MATCH (w1:Word {{key: "{current}"}})-[r:connects]-(w2:Word {{key: "{next}"}}) '
                         f'SET r.weight = {edges[edge]}')
            else:
                # Else, create it, with a starting weight of 1 meaning first co-occurence.
                edges[edge] = 1
                query = (f'MATCH (w1:Word {{key: "{current}"}}) '
                         f'MATCH (w2:Word {{key: "{next}"}}) '
                         f'MERGE (w1)-[r:connects {{weight: {edges[edge]}}}]-(w2)')
            # This line of code, is meant to be executed, in both cases of the if...else statement.
            database.execute(query, 'w')

    # Connect the paper, with all of its words.
    query = (f'MATCH (w:Word) WHERE w.key IN {terms} '
              'WITH collect(w) as words '
             f'MATCH (i:Issue {{key: "{filename}"}}) '
              'UNWIND words as word '
             f'CREATE (i)-[:{relationship}]->(word)')
    database.execute(query, 'w')
    return

def create_unique_constraints(database):
    """
    Wrapper function that gathers all CREATE CONSTRAINT queries,
    in one place.
    """
    database.execute('CREATE CONSTRAINT ON (word:Word) '
                     'ASSERT word.key IS UNIQUE', 'w')
    database.execute('CREATE CONSTRAINT ON (issue:Issue) '
                     'ASSERT issue.key IS UNIQUE', 'w')
    database.execute('CREATE CONSTRAINT ON (person:Person) '
                     'ASSERT person.uname IS UNIQUE', 'w')
    return

def create_issues_from_json(database, dirpath):
    """
    Function that creates the nodes representing issues,
    persons assigned to them, sets the properties of the
    first ones, and create the correspending graph of docs
    by using the title and description of the issue,
    based on the supplied json file.
    """
    current_system = platform.system()
    
    # Read json in memory.
    with open(dirpath, encoding = 'utf-8-sig', errors = 'ignore') as f:
        issues = json.load(f)['issues']

    skip_count = 0
    count = 1
    total_count = len(issues)

    # Process all issues.
    for issue in issues:
        # Print the number of the currently processed issue.
        print(f'Processing {count + skip_count} out of {total_count} issues...' )

        # Extract the title and description from the issue.
        title = '' if issue.get('title') is None else issue['title']
        description = '' if issue.get('description') is None else issue['description']

        # If the issue has no title and description, continue.
        if title == '' and description == '':
            skip_count += 1
            continue

        # Create the issue, using its fields.
        query = (
                f'CREATE (i:Issue {{key: "{issue["key"]}", '
                f'type: "{issue["type"]}", '
                f'priority: "{issue["priority"]}", '
                f'status: "{issue["status"]}"}})'
        )
        database.execute(query, 'w')

        # Create the assignee.
        query = (f'CREATE (p:Person {{uname: "{issue["assignee"]}"}})')
        database.execute(query, 'w')

        # Create the connection between the assignee and the issue.
        query = (
            f'MATCH (p:Person {{uname: "{issue["assignee"]}"}}) '
            f'MATCH (i:Issue {{key: "{issue["key"]}"}}) '
            f'CREATE (p)-[r:is_assigned_to]->(i)'
        )
        database.execute(query, 'w')

        # Join the text of the title and description.
        text = ' '.join((title, description))

        # Create the graph of words representation from the text of the issue.
        create_graph_of_words(generate_words(text), database, issue['key'], 'includes')

        # Update the progress counter.
        count = count + 1

        # Save the last accessed issue in a file.
        with open('last_accessed_issue.txt', 'w') as f:
            f.write(issue['key'])

        # Clear the screen to output the update the progress counter.
        clear_screen(current_system)

    print(f'Created {total_count - skip_count}, skipped {skip_count} issues.')
    return

def run_initial_algorithms(database):
    """
    Function that runs centrality & community detection algorithms,
    in order to prepare the data for analysis and visualization.
    Pagerank & Louvain are used, respectively.
    The calculated score for each node of the algorithms is being stored
    on the nodes themselves.
    """
    # Append the parameter 'weight' for the weighted version of the algorithm.
    pagerank(database, 'Word', 'connects', 20, 'pagerank')
    pagerank(database, 'Paper', 'cites', 20, 'pagerank')
    louvain(database, 'Word', 'connects', 'community')
    louvain(database, 'Paper', 'cites', 'community')
    return

def create_similarity_graph(database):
    """
    Function that creates a similarity graph
    based on Jaccard similarity measure.
    This measure connects the paper nodes with each other
    using the relationship 'is_similar', 
    which has the similarity score as a property.
    In order to prepare the data for analysis and visualization,
    we use Louvain Community detection algorithm.
    The calculated community id for each node is being stored
    on the nodes themselves.
    """
    # Remove similarity edges from previous iterations.
    database.execute('MATCH ()-[r:is_similar]->() DELETE r', 'w')

    # Create the similarity graph using Jaccard similarity measure.
    jaccard(database, 'Paper', 'includes', 'Word', 0.23, 'is_similar', 'score')

    # Find all similar document communities.
    # Append the parameter 'score' for the weighted version of the algorithm.
    louvain(database, 'Paper', 'is_similar', 'community')
    print('Similarity graph created.')
    return