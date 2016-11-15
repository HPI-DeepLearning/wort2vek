from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    Visualize word embeddings in a two dimensional space using t-SNE.
    """

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    pca = PCA(n_components=2)

    def __init__(self, model, decomposer=None):
        self.model = model
        if decomposer is None:
            self.decomposer = Visualizer.tsne
        else:
            self.decomposer = decomposer

    def embedding_size(self):
        return seld.model.syn0.shape[1]

    def get_embeddings(self, words):
        return self.model[words]

    def transform(self, embeddings):
        return self.decomposer.fit_transform(embeddings)

    def plot(self, two_d_embeddings, labels):
        plt.figure()
        for i, label in enumerate(labels):
            x, y = two_d_embeddings[i,:]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y))
        plt.show()

    def visualize(self, words):
        embeddings = self.get_embeddings(words)
        two_d_embeddings = self.transform(embeddings)
        self.plot(two_d_embeddings, words)
