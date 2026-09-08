from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('.github/agents/data/students.csv')


x = df[["hours_studied", "practice_tests"]]
y = df[["passed"]]

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)



print(accuracy)


