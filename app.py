from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)   # ✅ DO NOT override static_folder


def load_model_and_scaler(model_path, scaler_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    scaler = joblib.load(scaler_path)

    return (
        interpreter,
        scaler,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )


interpreter_lcoe, scaler_lcoe, in_lcoe, out_lcoe = load_model_and_scaler(
    "TRAINED MODEL FOR LCOE.tflite", "SCALER FOR LCOE.save"
)

interpreter_tes, scaler_tes, in_tes, out_tes = load_model_and_scaler(
    "TRAINED MODEL FOR ESC.tflite", "SCALER FOR ESC.save"
)


@app.route('/')
def index():
    return render_template('ui.html')   # ✅ FIX


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_choice = request.form['model_choice']
        features = [float(request.form[f'f{i+1}']) for i in range(9)]
        input_array = np.array([features], dtype=np.float32)

        if model_choice == 'LCOE':
            interpreter, scaler, input_details, output_details = (
                interpreter_lcoe, scaler_lcoe, in_lcoe, out_lcoe
            )
        elif model_choice == 'TES':
            interpreter, scaler, input_details, output_details = (
                interpreter_tes, scaler_tes, in_tes, out_tes
            )
        else:
            return jsonify({'error': 'Invalid model choice'}), 400

        input_scaled = scaler.transform(input_array).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        return jsonify({'prediction': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
