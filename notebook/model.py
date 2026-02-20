import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# load data
batch = os.getcwd()
path = os.path.join(batch, '..', 'data/student_prediction.csv')
df = pd.read_csv(path)

X = df.drop("final_score", axis=1)
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    
    n_estimators=400,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

model_path = os.path.join(batch, '..', 'model.pkl')

joblib.dump(model, model_path)
print("Model saved successfully")
