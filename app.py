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
        runtime = boto3.client('sagemaker-runtime')
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

def call_translation_api(text, target_language='fr', decode_type='beam', beam_size=5):
    payload = {
        'text': text,
        'target_language' : target_language,
        'decoder_type': decode_type,
        'beam_size':beam_size
    }
    try:
        response = requests.post(VM_API_URL, json=payload)
        response.raise_for_status()
        json_response = response.json()
        #print(json_response['attentions'][0][0])
        return (json_response['translation'][0][0], json_response['input_tokens'][0],
                json_response['output_tokens'][0], json_response['attentions'][0][0])
    except requests.HTTPError as http_err:
        print("HTTP error occurred:", http_err)
        print("Response content:", response.text)
        raise
    except Exception as e:
        print("Other error occurred:", e)
        raise


@app.route('/projects/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        text = request.form.get('source_text', '').strip()
        target_language = request.form.get('target_language', 'fr')
        decode_type = request.form.get('decode_type', 'beam')
        beam_size = int(request.form.get('beam_size', 5))

        if not text:
            return render_template('translate.html', error="Please enter a sentence.")  
        try:
            results, input_tokens, output_tokens, attention_matrix = call_translation_api([text],
                                                                                    target_language=target_language,
                                                                                    decode_type=decode_type,
                                                                                    beam_size=beam_size)
            #print("attention: ", attention_matrix, flush=True)
            return render_template('translate.html', source_text=text, translation=results,
                                   input_tokens=input_tokens, output_tokens=output_tokens,
                                   attentions=attention_matrix,
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