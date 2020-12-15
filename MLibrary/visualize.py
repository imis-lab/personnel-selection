import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

model = Word2Vec.load('jira_issues_large.model')

NUMBER_OF_WORDS = 500
PCA_N_COMPONENTS = 50
TSNE_PERPLEXITY = 30

data = []
for key in model.wv.vocab.keys():
    data.append({'word': key, 'embedding': model.wv[key]})

embedding_df = pd.DataFrame(data)
embedding_df = pd.DataFrame(embedding_df.iloc[:NUMBER_OF_WORDS])

embedding_df['pca'] = PCA(n_components=PCA_N_COMPONENTS).fit_transform(embedding_df['embedding'].to_list()).tolist()
tsne_values = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY).fit_transform(embedding_df['pca'].to_list())
embedding_df['tsne_x'] = tsne_values[:, 0]
embedding_df['tsne_y'] = tsne_values[:, 1]

import plotly.express as px

fig = px.scatter(embedding_df, x="tsne_x", y="tsne_y", text="word")
fig.update_traces(textposition='top center')
fig.update_layout(height=1000)
fig.show()
