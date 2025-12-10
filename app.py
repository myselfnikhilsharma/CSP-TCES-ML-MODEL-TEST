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

model_names = [
    "Overall LCOE",
    "Collector contribution in LCOE",
    "O&M contribution in LCOE",
    "Power cycle contribution in LCOE",
    "Receiver contribution in LCOE",
    "Storage contribution in LCOE"
]


# Load one or more ESC/TES models. To add more ESC models, add additional
# calls to `load_model_and_scaler` here and append names to `tes_model_names`.
tes_models = [
    load_model_and_scaler("TRAINED MODEL FOR ESC.tflite", "SCALER FOR ESC.save"),
    load_model_and_scaler("TRAINED MODEL FOR ESC HEAT EXCHANGER.tflite", "SCALER FOR ESC HEAT EXCHANGER.save"),
    load_model_and_scaler("TRAINED MODEL FOR ESC REACTOR.tflite", "SCALER FOR ESC REACTOR.save"),
    load_model_and_scaler("TRAINED MODEL FOR ESC SOLID STORAGE TANK.tflite", "SCALER FOR ESC SOLID STORAGE TANK.save"),
    load_model_and_scaler("TRAINED MODEL FOR ESC STORAGE MATERIAL.tflite", "SCALER FOR ESC STORAGE MATERIAL.save")
]

tes_model_names = ["Overall ESC",
                   "ESC Heat Exchanger",
                   "ESC Reactor",
                   "ESC Solid Storage Tank",
                   "ESC Storage Material"
]


@app.route('/')
def index():
    return send_from_directory('', 'UI.html')




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -----------------------------
        # SAFE conversion of all inputs
        # -----------------------------
        features = []
        for i in range(9):
            raw = request.form.get(f'f{i+1}', "")

            # Remove commas, spaces
            cleaned = raw.replace(",", "").strip()

            try:
                value = float(cleaned)
                features.append(value)
            except ValueError:
                return jsonify({
                    'error': f"Invalid number in input f{i+1}: '{raw}'"
                }), 400

        input_array = np.array([features], dtype=np.float32)
        # -------------------------
        # LCOE → MULTIPLE MODELS (run all and return)
        # -------------------------
        results = []

        for interpreter, scaler, input_details, output_details in lcoe_models:
            input_scaled = scaler.transform(input_array).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_scaled)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
            results.append(round(float(pred), 2))

        named_results = {model_names[i]: round(float(results[i]), 2) for i in range(len(results))}

        # -------------------------
        # ESC/TES → run all ESC models and return list
        # -------------------------
        esc_results = []
        for (interpreter, scaler, in_details, out_details) in tes_models:
            input_scaled_tes = scaler.transform(input_array).astype(np.float32)
            interpreter.set_tensor(in_details[0]['index'], input_scaled_tes)
            interpreter.invoke()
            tes_prediction = interpreter.get_tensor(out_details[0]['index'])[0][0]
            esc_results.append(round(float(tes_prediction), 2))

        return jsonify({'predictions': named_results, 'esc': esc_results, 'esc_model_names': tes_model_names})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)

