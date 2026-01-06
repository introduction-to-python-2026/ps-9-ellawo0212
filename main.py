import pandas as pd
df = pd.read_csv("parkinsons.csv")
df = df.dropna()
sf=[ 'spread1', 'spread2']
x = df[sf]
y = df["status"]
from sklearn.preprocessing import MinMaxScaler
Scalar=MinMaxScaler()
x=Scalar.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test =train_test_split(x,y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
KN_model=KNeighborsClassifier(n_neighbors=30)
KN_model.fit(x_train , y_train)
from sklearn.metrics import accuracy_score
y_pred = KN_model.predict(x_test)
accuracy= accuracy_score(y_test, y_pred)
print("accuracy:",accuracy)
import joblib
joblib.dump(KN_model, 'my_model.joblib')
