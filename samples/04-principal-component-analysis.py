import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; 
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

sns.set()
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');
plt.show()

pca = PCA(n_components=2)
pca.fit(X)
print('pca components:', pca.components_)
print('pca explained variance:', pca.explained_variance_)

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()

digits = load_digits()
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_rainbow_r',10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()