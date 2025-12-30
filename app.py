from flask import Flask, request, jsonify, send_from_directory 
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__, static_folder='')

# ========================================
# LCOE MODELS - FIXED
# ========================================
lcoe_interpreters = []
lcoe_scalers = []
lcoe_input_details = []
lcoe_output_details = []

lcoe_paths = [
    ("TRAINED MODEL FOR LCOE.tflite", "SCALER FOR LCOE.save"),
    ("TRAINED MODEL FOR LCOE COLLECTOR.tflite", "SCALER FOR LCOE COLLECTOR.save"),
    ("TRAINED MODEL FOR LCOE OPERATION AND MANAGEMENT.tflite", "SCALER FOR LCOE OPERATION AND MANAGEMENT.save"),
    ("TRAINED MODEL FOR LCOE POWER CYCLE.tflite", "SCALER FOR LCOE POWER CYCLE.save"),
    ("TRAINED MODEL FOR LCOE RECEIVER.tflite", "SCALER FOR LCOE RECEIVER.save"),
    ("TRAINED MODEL FOR LCOE STORAGE.tflite", "SCALER FOR LCOE STORAGE.save")
]

for model_path, scaler_path in lcoe_paths:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    scaler = joblib.load(scaler_path)
    lcoe_interpreters.append(interpreter)
    lcoe_scalers.append(scaler)
    lcoe_input_details.append(interpreter.get_input_details())
    lcoe_output_details.append(interpreter.get_output_details())

model_names = [
    "Overall LCOE",
    "Collector contribution in LCOE",
    "O&M contribution in LCOE",
    "Power cycle contribution in LCOE",
    "Receiver contribution in LCOE",
    "Storage contribution in LCOE"
]

# ========================================
# TES/ESC MODELS - FIXED
# ========================================
tes_interpreters = []
tes_scalers = []
tes_input_details = []
tes_output_details = []

tes_paths = [
    ("TRAINED MODEL FOR ESC.tflite", "SCALER FOR ESC.save"),
    ("TRAINED MODEL FOR ESC HEAT EXCHANGER.tflite", "SCALER FOR ESC HEAT EXCHANGER.save"),
    ("TRAINED MODEL FOR ESC REACTOR.tflite", "SCALER FOR ESC REACTOR.save"),
    ("TRAINED MODEL FOR ESC SOLID STORAGE TANK.tflite", "SCALER FOR ESC SOLID STORAGE TANK.save"),
    ("TRAINED MODEL FOR ESC STORAGE MATERIAL.tflite", "SCALER FOR ESC STORAGE MATERIAL.save")
]

for model_path, scaler_path in tes_paths:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    scaler = joblib.load(scaler_path)
    tes_interpreters.append(interpreter)
    tes_scalers.append(scaler)
    tes_input_details.append(interpreter.get_input_details())
    tes_output_details.append(interpreter.get_output_details())

tes_model_names = ["Overall ESC",
                   "ESC Heat Exchanger",
                   "ESC Reactor",
                   "ESC Solid Storage Tank",
                   "ESC Storage Material"]



lcoe_feature_indices = []
for scaler in lcoe_scalers:
    n_features = scaler.n_features_in_
    lcoe_feature_indices.append(list(range(n_features)))

tes_feature_indices = []
for scaler in tes_scalers:
    n_features = scaler.n_features_in_
    tes_feature_indices.append(list(range(n_features)))

print("LCOE models expect:", [len(idx) for idx in lcoe_feature_indices])
print("TES models expect:", [len(idx) for idx in tes_feature_indices])

@app.route('/')
def index():
    return send_from_directory('', 'UI.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for i in range(9):
            raw = request.form.get(f'f{i+1}', "")
            cleaned = raw.replace(",", "").strip()
            try:
                value = float(cleaned)
                features.append(value)
            except ValueError:
                return jsonify({
                    'error': f"Invalid number in input f{i+1}: '{raw}'"
                }), 400

        input_array = np.array([features], dtype=np.float32)
        
        # LCOE predictions - FIXED WITH SLICING
        results = []
        for i in range(len(lcoe_interpreters)):
            model_indices = lcoe_feature_indices[i]  # ← USE THE INDICES!
            X_subset = input_array[:, model_indices]  # ← SLICE FEATURES!
            input_scaled = lcoe_scalers[i].transform(X_subset).astype(np.float32)
            lcoe_interpreters[i].set_tensor(lcoe_input_details[i][0]['index'], input_scaled)
            lcoe_interpreters[i].invoke()
            pred = lcoe_interpreters[i].get_tensor(lcoe_output_details[i][0]['index'])[0][0]
            results.append(round(float(pred), 2))

        named_results = {model_names[i]: results[i] for i in range(len(results))}

        # ESC/TES predictions - FIXED WITH SLICING
        esc_results = []
        for i in range(len(tes_interpreters)):
            model_indices = tes_feature_indices[i]  # ← USE THE INDICES!
            X_subset = input_array[:, model_indices]  # ← SLICE FEATURES!
            input_scaled_tes = tes_scalers[i].transform(X_subset).astype(np.float32)
            tes_interpreters[i].set_tensor(tes_input_details[i][0]['index'], input_scaled_tes)
            tes_interpreters[i].invoke()
            tes_prediction = tes_interpreters[i].get_tensor(tes_output_details[i][0]['index'])[0][0]
            esc_results.append(round(float(tes_prediction), 2))

        return jsonify({'predictions': named_results, 'esc': esc_results, 'esc_model_names': tes_model_names})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

