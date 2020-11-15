import seaborn as sns
from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import datasets

iris = sns.load_dataset('iris')

sns.set()
# sns.pairplot(iris,hue='species')
# plt.show()

X_iris = iris.drop('species', axis=1)
y_iris = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [value for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.5f%%" % (accuracy * 100.0))

# sklearn datasets
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# transform to DMatrix
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 100  # The number of training iterations

model = xgb.train(param, D_train, steps)

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision: {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall: {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy: {}".format(accuracy_score(y_test, best_preds)))
