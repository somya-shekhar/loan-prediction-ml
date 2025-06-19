from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Adjust model path
MODEL_PATH = os.path.join("model", "final_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            input_data = {
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
                "Property_Area": [request.form["Property_Area"]],
            }

            df = pd.DataFrame(input_data)
            pred = model.predict(df)[0]
            result = "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("form.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
