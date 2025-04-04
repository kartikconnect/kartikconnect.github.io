from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your ML model (Titanic Survival Prediction)
model = pickle.load(open("titanic_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/projects")
def projects():
    return render_template("projects.html")

@app.route("/ml-demo", methods=["GET", "POST"])
def ml_demo():
    prediction = None
    if request.method == "POST":
        # Example: collect input fields
        age = float(request.form["age"])
        fare = float(request.form["fare"])
        sex = 1 if request.form["sex"] == "male" else 0
        pclass = int(request.form["pclass"])

        features = np.array([[pclass, sex, age, fare]])
        prediction = model.predict(features)[0]

    return render_template("ml_demo.html", prediction=prediction)

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
