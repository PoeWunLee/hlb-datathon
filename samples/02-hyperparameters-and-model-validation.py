from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X1, y1)
y2_model = model.predict(X2)
accuracy = accuracy_score(y2, y2_model)
print(accuracy)