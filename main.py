import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
df = pd.read_csv("parkinsons.csv")
df = df.dropna()
selected_features = ['spread1', 'spread2']  
x = df[selected_features]
y = df["status"]  
KN_model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=60))
])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
KN_model.fit(x_train, y_train)
Y_pred = KN_model.predict(x_test)
accuracy = accuracy_score(y_test, Y_pred)
print("accuracy:", accuracy)
joblib.dump(KN_model, 'my_model.joblib')
])
KN_model=KNeighborsClassifier(n_neighbors=30)
KN_model.fit(x_train , y_train)
Y_pred = KN_model.predict(x_test)
accuracy= accuracy_score(y_test, Y_pred)
print("accuracy:",accuracy)
import joblib
joblib.dump(KN_model, 'my_model.joblib')
