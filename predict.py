import mlflow
import pandas as pd

#ZADANIE 5

mlflow.set_tracking_uri("http://localhost:5000")

RUN_ID = "33b2d3be3e084a6bbff09b8e2f917494"

loaded_model  = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

sample = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,
    "Age": 25,
    "Fare": 50.0
}])

print(f"\nPrzykładowa próbka: \n{sample}")

prediction = loaded_model.predict(sample)
proba = loaded_model.predict_proba(sample)

print(f"\nPrzewidywana klasa (Survived): "
      f"\n{int(prediction[0])}"
      f"\n\nPrawdopodobieństwa klas [nie przeżył, przeżył]: \n{proba[0]}")