from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from landmarks import *
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

process_dataset("dataset", "landmarks.csv")
x, y = read_landmarks_from_csv("landmarks.csv")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
process_dataset("dataset_val", "landmarks_val.csv")
x_val, y_val = read_landmarks_from_csv("landmarks_val.csv")

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}

for name, model in models.items():
    model.fit(x_train, y_train)
    test_score = accuracy_score(y_test, model.predict(x_test))
    print(f"{name} Test Score: {test_score*100}")
    scores = cross_val_score(model, x_val, y_val, cv=5)
    print(f"{name} Validation Score: {np.mean(scores)*100}")
