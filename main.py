import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data = pd.read_csv("parkinsons.csv")

X = data[config["features"]]
y = data["status"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model = LogisticRegression(solver="liblinear", max_iter=1000)
model.fit(X, y)

joblib.dump(model, config["path"])
