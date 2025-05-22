from flask import Flask, render_template, request, jsonify
import os
import boto3
import json
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

ENDPOINT_NAME = 'ml-model-endpoint'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/capstone')
def capstone():
    return render_template('capstone.html')

@app.route('/paper')
def paper():
    return render_template('paper.html')

@app.route('/projects/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4'])
            ]
            logging.info(f"Received features: {features}")
        except ValueError:
            logging.warning("Invalid input submitted")
            return 'Invalid input. Please enter valid numebers.'
    
        # send to SageMaker
        runtime = boto3.client('sagemaker-runtime')
        payload = json.dumps({'data': [features]})
        logging.info("Sending payload to SageMaker")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        result = json.loads(response['Body'].read().decode())
        prediction = result['predictions'][0]
        logging.info(f"Prediction result: {prediction}")

        return render_template('predict.html', prediction=prediction, features=features)
    return render_template('predict.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render will set PORT env var
    app.run(host='0.0.0.0', port=port, debug=True)