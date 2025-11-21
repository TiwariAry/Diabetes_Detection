import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
CUSTOM_THRESHOLD = 0.40 # Optimal Threshold
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction', 'Age']

try:
    with open('logreg_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('imputation_medians.pkl', 'rb') as medians_file:
        imputation_medians = pickle.load(medians_file)
    print("Model, Scaler, and Medians loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Missing file {e.filename}. Ensure all .pkl files are in the directory.")
    model, scaler, imputation_medians = None, None, None


def preprocess_input(input_data):
    data_dict = {name: [float(input_data[name])] for name in FEATURE_NAMES}
    input_df = pd.DataFrame(data_dict, columns=FEATURE_NAMES)

    for col, median_val in imputation_medians.items():
        if col in input_df.columns:
            if input_df[col].iloc[0] == 0:
                input_df[col].iloc[0] = median_val

    scaled_data = scaler.transform(input_df)

    return scaled_data


@app.route('/', methods=['GET', 'POST'])
def predict_page():
    prediction_result = None

    if request.method == 'POST':
        if model is None or scaler is None:
            return render_template('index.html', error='Model/Scaler not loaded.'), 500

        try:
            form_data = request.form.to_dict()

            scaled_input = preprocess_input(form_data)

            probability = model.predict_proba(scaled_input)[:, 1][0]

            prediction = 1 if probability >= CUSTOM_THRESHOLD else 0

            prediction_result = {
                'prediction': prediction,
                'probability_of_diabetes': f"{probability:.4f}",
                'outcome_message': 'POSITIVE (Diabetic)' if prediction == 1 else 'NEGATIVE (Non-Diabetic)',
                'risk_level': 'HIGH RISK - Follow-up needed' if prediction == 1 else 'Low Risk'
            }

        except Exception as error:
            prediction_result = {'error': f"An error occurred: {str(error)}"}

    return render_template('index.html', result=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)