import pandas as pd

df = pd.read_csv("/content/parkinsons.csv")
df = df.dropna()

print(df.columns.tolist())

selected_features = ['RPDE', 'PPE']
X = df[selected_features]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(max_depth=3)
DTC.fit(X_train, y_train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

from sklearn.metrics import accuracy_score
y_pred = DTC.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
