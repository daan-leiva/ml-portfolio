from flask import Flask, render_template, request, jsonify
import os
import boto3
import json
import logging
import time
import requests

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

endpoint_cache = {
    'status': None,
    'last_checked': 0
}

ENDPOINT_NAME = 'ml-model-endpoint'
VM_API_URL = 'http://35.212.198.98:5000/translate'

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
    print('aaaa')
    if request.method == 'POST':
        if not is_model_ready():
            return render_template('predict.html', error="Model is not ready. Please try again later.")
        try:
            features = [
                float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4'])
            ]
        except ValueError:
            logging.error("Invalid input submitted")
            return 'Invalid input. Please enter valid numebers.'
    
        # send to SageMaker
        runtime = boto3.client      ('sagemaker-runtime')
        payload = json.dumps({'data': [features]})
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        result = json.loads(response['Body'].read().decode())
        prediction = result['predictions'][0]

        return render_template('predict.html', prediction=prediction, features=features)
    return render_template('predict.html')

def call_translation_api(text, decode_type='beam', beam_size=5):
    payload = {
        'text': text,
        'decoder_type': decode_type,
        'beam_size':beam_size
    }
    try:
        response = requests.post(VM_API_URL, json=payload)
        response.raise_for_status()
        print(response)
        return response.json()['translation'][0]
    
    except requests.RequestException as e:
        return f"API error: {e}"

@app.route('/projects/translate', methods=['GET', 'POST'])
def translate():
    print('aaaaaaaaaaaa')
    if request.method == 'POST':
        text = request.form.get('source_text', '').strip()
        decode_type = request.form.get('decode_type', 'beam')
        beam_size = int(request.form.get('beam_size', 5))

        if not text:
            return render_template('translate.html', error="Please enter a sentence.")
        
        try:
            results = call_translation_api([text], decode_type=decode_type, beam_size=beam_size)
            print(results)
            return render_template('translate.html', original=text, translated=results,
                                   decode_type=decode_type, beam_size=beam_size)
        except Exception as e:
            return render_template('translate.html', error=str(e))

    return render_template('translate.html')

def is_model_ready():
    now = time.time()
    if now - endpoint_cache['last_checked'] > 60: # once per minute
        sm_client = boto3.client('sagemaker')
        status = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)['EndpointStatus']
        endpoint_cache['status'] = status
        endpoint_cache['last_checked'] = now
    return endpoint_cache['status'] == 'InService'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render will set PORT env var
    app.run(host='0.0.0.0', port=port, debug=True)