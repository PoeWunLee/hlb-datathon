from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

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
print(predictions)
print(y_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.5f%%" % (accuracy * 100.0))