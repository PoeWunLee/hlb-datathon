from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X1, y1)
    y2_model = model.predict(X2)
    accuracy = accuracy_score(y2, y2_model)
    print('cross-validation', accuracy)

    # five-fold cross-validation
    five_fold_score = cross_val_score(model, X, y, cv=5)
    print('five-fold cross-val:', five_fold_score)

    # polynomial regression
    X_np, y_np = make_data(40)
    sns.set()  # plot formatting
    X_test = np.linspace(-0.1, 1.1, 500)[:, None]

    plt.scatter(X_np.ravel(), y_np, color='black')
    axis = plt.axis()
    for degree in [1, 3, 5]:
        y_test = PolynomialRegression(degree).fit(X_np, y_np).predict(X_test)
        plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
    plt.xlim(-0.1, 1.0)
    plt.ylim(-2, 12)
    plt.legend(loc='best')
    plt.show()

    # grid search
    param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}

    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
    grid.fit(X_np, y_np)
    print(grid.best_params_)

    model = grid.best_estimator_

    plt.scatter(X_np.ravel(), y_np)
    lim = plt.axis()
    y_test = model.fit(X_np, y_np).predict(X_test)
    plt.plot(X_test.ravel(), y_test)
    plt.axis(lim)
    plt.show()

if __name__ == '__main__':
    main()