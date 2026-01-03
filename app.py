from flask import Flask, request, jsonify, send_from_directory 
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__, static_folder='')

# Load LCOE model
lcoe_interpreter = tf.lite.Interpreter(model_path="TRAINED MODEL FOR LCOE.tflite")
lcoe_interpreter.allocate_tensors()
lcoe_scaler = joblib.load("SCALER FOR LCOE.save")
lcoe_input_details = lcoe_interpreter.get_input_details()
lcoe_output_details = lcoe_interpreter.get_output_details()

# Load ESC model
esc_interpreter = tf.lite.Interpreter(model_path="TRAINED MODEL FOR ESC.tflite")
esc_interpreter.allocate_tensors()
esc_scaler = joblib.load("SCALER FOR ESC.save")
esc_input_details = esc_interpreter.get_input_details()
esc_output_details = esc_interpreter.get_output_details()

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
        
        # LCOE prediction
        input_scaled = lcoe_scaler.transform(input_array).astype(np.float32)
        lcoe_interpreter.set_tensor(lcoe_input_details[0]['index'], input_scaled)
        lcoe_interpreter.invoke()
        lcoe_pred = lcoe_interpreter.get_tensor(lcoe_output_details[0]['index'])[0][0]
        
        # ESC prediction
        input_scaled_esc = esc_scaler.transform(input_array).astype(np.float32)
        esc_interpreter.set_tensor(esc_input_details[0]['index'], input_scaled_esc)
        esc_interpreter.invoke()
        esc_pred = esc_interpreter.get_tensor(esc_output_details[0]['index'])[0][0]

        return jsonify({
            'predictions': {'LCOE': round(float(lcoe_pred), 4)},
            'esc': [round(float(esc_pred), 4)]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

