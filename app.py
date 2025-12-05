from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
import tensorflow as tf


app = Flask(__name__, static_folder='')


# Function to load model + scaler once
def load_model_and_scaler(model_path, scaler_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    scaler = joblib.load(scaler_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, scaler, input_details, output_details


# Load MULTIPLE LCOE models
lcoe_models = [
    load_model_and_scaler("TRAINED MODEL FOR LCOE.tflite", "SCALER FOR LCOE.save"),
    load_model_and_scaler("TRAINED MODEL FOR LCOE COLLECTOR.tflite", "SCALER FOR LCOE COLLECTOR.save"),
    load_model_and_scaler("TRAINED MODEL FOR LCOE OPERATION AND MANAGEMENT.tflite", "SCALER FOR LCOE OPERATION AND MANAGEMENT.save"),
    load_model_and_scaler("TRAINED MODEL FOR LCOE POWER CYCLE.tflite", "SCALER FOR LCOE POWER CYCLE.save"),
    load_model_and_scaler("TRAINED MODEL FOR LCOE RECEIVER.tflite", "SCALER FOR LCOE RECEIVER.save"),
    load_model_and_scaler("TRAINED MODEL FOR LCOE STORAGE.tflite", "SCALER FOR LCOE STORAGE.save"),
]


interpreter_tes, scaler_tes, in_tes, out_tes = load_model_and_scaler(
    "TRAINED MODEL FOR ESC.tflite", "SCALER FOR ESC.save"
)


@app.route('/')
def index():
    return send_from_directory('', 'ui.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_choice = request.form['model_choice']  # "LCOE" or "TES"

        # Collect features
        features = [float(request.form[f'f{i+1}']) for i in range(9)]
        input_array = np.array([features], dtype=np.float32)


        # -------------------------
        # LCOE → Run MULTIPLE MODELS
        # -------------------------
        # Pick correct model
        if model_choice == 'LCOE':
            results = []

            for interpreter, scaler, input_details, output_details in lcoe_models:
                # Scale input
                input_scaled = scaler.transform(input_array).astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], input_scaled)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
                results.append(round(float(pred), 2))

            return jsonify({'predictions': results})

        # -------------------------
        # TES → Single Model
        # -------------------------
        elif model_choice == 'TES':
            input_scaled = scaler_tes.transform(input_array).astype(np.float32)

            interpreter_tes.set_tensor(in_tes[0]['index'], input_scaled)
            interpreter_tes.invoke()

            prediction = interpreter_tes.get_tensor(out_tes[0]['index'])[0][0]

            return jsonify({'prediction': round(float(prediction), 2)})
        
        else:
            return jsonify({'error': 'Invalid model choice'}), 400


    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)