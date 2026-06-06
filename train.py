import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

#ZADANIA 2 i 3

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


max_depth_values = [3, 5, 7]

for max_depth in max_depth_values:
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators = 100, max_depth = max_depth, random_state = 42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.set_tracking_uri("http://localhost:5000")

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, artifact_path = "model")




