from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def tsne_plot(n_components=2):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    with open("glove.6B.100d.txt", "r") as f:
        i = 0
        for line in f:
            splited_line = line.split()
            label, token = splited_line[0], splited_line[1:]
            labels.append(label)
            tokens.append(np.asarray(token))
            i+=1
            if i>100: break

    tsne_model = TSNE(perplexity=40, n_components=n_components, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    fig = plt.figure(figsize=(16, 16))
    if n_components==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(new_values[:,0],new_values[:,1],new_values[:,2],c="r",marker="o")
        for i in range(len(new_values)):
            ax.text(new_values[i][0],new_values[i][1],new_values[i][2],labels[i])
    else:
        plt.scatter(new_values[:,0],new_values[:,1])
        for i in range(len(new_values)):
            plt.annotate(labels[i],
                        xy=(new_values[i][0],new_values[i][1]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        plt.savefig('sample.png', bbox_inches="tight")
    return new_values,labels


new_values,labels = tsne_plot(n_components=2)
