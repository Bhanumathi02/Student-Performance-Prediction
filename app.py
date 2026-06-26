from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])

    prediction = model.predict([[study_hours, attendance, previous_marks]])

    if prediction[0] == 1:
        result = "Excellent Performance 🎉"
    else:
        result = "Needs Improvement 📚"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)