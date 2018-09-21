import numpy as np 
import sklearn.manifold
from matplotlib import pyplot as plt 
from dependencyRNN import DependencyRNN

random = DependencyRNN.load('random_initz.npz')
keys = random.answers.keys()
X = random.answers.values()

x = []
y = []

tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0)
X_reduced = tsne.fit_transform(X)

for i in range(0, len(keys)):
	x.append(X_reduced[i][0])
	y.append(X_reduced[i][1])

plt.scatter(x,y)
for i in range(0, len(keys)):
	plt.annotate(keys[i], xy=(x[i], y[i]), fontsize=8)

plt.show()