import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

iris = sns.load_dataset('iris')

sns.set()
sns.pairplot(iris,hue='species')
plt.show()
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)
# instantiate model
model_naive_bayes = GaussianNB()
# fit model to data
model_naive_bayes.fit(Xtrain, ytrain)
# predict on new data
y_model = model_naive_bayes.predict(Xtest)
iris_accuracy = accuracy_score(ytest, y_model)
print("iris accuracy:", iris_accuracy)

# instantiate the model with hyperparameters
model_pca = PCA(n_components=2)
# fit to data, notice y is not specified
model_pca.fit(X_iris)
# transform the data to two dimensions
X_2D = model_pca.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
plt.show()

# instantiate the model with hyperparameters
model_gmm = GaussianMixture(n_components=3, covariance_type='full')  
# fit to data, y is not specified
model_gmm.fit(X_iris)
# determine cluster labels
y_gmm = model_gmm.predict(X_iris)
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species', col='cluster', fit_reg=False);
plt.show()

# linear regression                                             
rng = np.random.RandomState(42)
x = 10*rng.rand(50)
y = 2*x-1+rng.randn(50)
model = LinearRegression(fit_intercept=True)
X = x.reshape(x.shape[0],1)
model.fit(X, y)

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()