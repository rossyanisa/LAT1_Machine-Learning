from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = r"model\hasil_pelatihan_model.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as model_file:
            ml_model = joblib.load(model_file)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.route("/")
def home():
    # Ensure the template file exists
    template_path = os.path.join(app.root_path, 'templates', 'home.html')
    if os.path.exists(template_path):
        return render_template('home.html')
    else:
        return f"Template not found: {template_path}"

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("Prediction started")

    if request.method == 'POST':
        try:
            # Extract values from the form
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])

            # Prepare the data for the model
            pred_args = [RnD_Spend, Admin_Spend, Market_Spend]
            pred_args_arr = np.array(pred_args).reshape(1, -1)

            # Perform prediction
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)

        except ValueError as ve:
            return f"Invalid input data: {str(ve)}"
        except FileNotFoundError as fnf_error:
            return f"Model file not found: {str(fnf_error)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

        # Ensure the template for prediction exists
        predict_template_path = os.path.join(app.root_path, 'templates', 'predict.html')
        if os.path.exists(predict_template_path):
            return render_template('predict.html', prediction=model_prediction)
        else:
            return f"Prediction template not found: {predict_template_path}"

if __name__ == "__main__":
    # Debug mode for development
    app.run(host='0.0.0.0', port=5000, debug=True)
