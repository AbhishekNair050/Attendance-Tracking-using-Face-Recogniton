from landmarks import process_dataset, read_landmarks_from_csv
from sklearn.ensemble import RandomForestClassifier
import pickle

process_dataset("dataset", "landmarks.csv")
print("INFO - Dataset processed")
x, y = read_landmarks_from_csv("landmarks.csv")
print("INFO - Landmarks read from CSV")
rf = RandomForestClassifier()
rf.fit(x, y)
print("INFO - Random Forest model trained")
with open("utils/model.pkl", "wb") as file:
    pickle.dump(rf, file)

model = pickle.load(open("utils/model.pkl", "rb"))
print(model.score(x, y))
