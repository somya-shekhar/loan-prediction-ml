from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the pipeline model
MODEL_PATH = os.path.join("model", "loan_model.joblib")  # Adjust if model path differs
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Home route with form
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_result = None
    if request.method == "POST":
        try:
            data = {
                "Gender": [request.form["Gender"]],
                "Married": [request.form["Married"]],
                "Dependents": [request.form["Dependents"]],
                "Education": [request.form["Education"]],
                "Self_Employed": [request.form["Self_Employed"]],
                "ApplicantIncome": [int(request.form["ApplicantIncome"])],
                "CoapplicantIncome": [float(request.form["CoapplicantIncome"])],
                "LoanAmount": [float(request.form["LoanAmount"])],
                "Loan_Amount_Term": [float(request.form["Loan_Amount_Term"])],
                "Credit_History": [int(request.form["Credit_History"])],
                "Property_Area": [request.form["Property_Area"]]
            }

            df = pd.DataFrame(data)
            prediction = model.predict(df)[0]
            prediction_result = "Approved ✅" if prediction == 1 else "Rejected ❌"
        except Exception as e:
            prediction_result = f"Error during prediction: {e}"

    return render_template("form.html", result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
