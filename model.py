import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/student_data.csv")

# Input (X) and Output (y)
X = data[["StudyHours", "Attendance", "PreviousMarks"]]
y = data["Performance"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Save the model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")