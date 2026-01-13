import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data = pd.read_csv("parkinsons.csv")

if "features" in config:
    X = data[config["features"]]
else:
    X = data[config["selected_features"]]

y = data["status"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(solver="liblinear", max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(model, config["path"])

