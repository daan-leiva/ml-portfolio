from flask import Flask, render_template, request, jsonify
import os
import boto3
import json
import logging
import time
import requests
import base64

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

endpoint_cache = {
    'status': None,
    'last_checked': 0
}

ENDPOINT_NAME = 'ml-model-endpoint'
VM_API_URL = 'http://35.212.229.126:5000/translate'
EC3_MEDIMG_API_URL = 'http://54.245.90.136:5000/predict_medimg'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/resume')
def resume():
    return render_template('resume.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            # Extract form data
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            message = request.form.get('message', '').strip()

            # Basic validation
            if not name or not email or not message:
                return render_template('contact.html', error="All fields are required.")

            # Prepare entry
            entry = {
                'name': name,
                'email': email,
                'message': message,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save to local JSON file
            filepath = 'data/contact_messages.json'
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if os.path.exists(filepath):
                with open(filepath, 'r+') as f:
                    messages = json.load(f)
                    messages.append(entry)
                    f.seek(0)
                    json.dump(messages, f, indent=2)
            else:
                with open(filepath, 'w') as f:
                    json.dump([entry], f, indent=2)

            return render_template('contact.html', success="Message submitted successfully!")
        
        except Exception as e:
            return render_template('contact.html', error=f"An error occurred: {str(e)}")

    return render_template('contact.html')



@app.route('/capstone')
def capstone():
    return render_template('capstone.html')

@app.route('/paper')
def paper():
    return render_template('paper.html')

@app.route('/projects/predict', methods=['GET', 'POST'])
def predict():
    # Handle POST request with prediction input
    if request.method == 'POST':
        # Ensure the SageMaker endpoint is live before proceeding
        # deprecated when moved to serverless
        #if not is_model_ready():
            #return render_template('predict.html', error="Model is not ready. Please try again later.")
        try:
            # Parse and convert form inputs to floats
            features = [
                float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4'])
            ]
        except ValueError:
            # Log and return an error if input values are invalid
            logging.error("Invalid input submitted")
            return 'Invalid input. Please enter valid numbers.'
        
        # Map index to flower name
        label_map = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }
        
        # Prepare SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime')
        
        # Construct JSON payload with user inputs
        payload = json.dumps({'data': [features]})
        
        # Send request to deployed SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse prediction result from response
        result = json.loads(response['Body'].read().decode())
        predicted_index = result['predictions'][0]
        prediction = label_map.get(predicted_index, f"Unknown ({predicted_index})")

        # Render the template with prediction result and input features
        return render_template('predict.html', prediction=prediction, features=features)
    
    # Render empty form for GET requests
    return render_template('predict.html')


# Calls the translation model API running on the VM with provided input text and decoding options.
# Returns translated text, input/output tokens, and the attention matrix for visualization.
def call_translation_api(text, target_language='fr', decode_type='beam', beam_size=5):
    # Prepare the payload for the API request
    payload = {
        'text': text,
        'target_language': target_language,
        'decoder_type': decode_type,
        'beam_size': beam_size
    }

    try:
        # Send request to the translation API endpoint
        response = requests.post(VM_API_URL, json=payload)

        # Raise exception if HTTP response code indicates error
        response.raise_for_status()

        # Parse JSON response from the API
        json_response = response.json()

        # Extract and return translation results
        return (
            json_response['translation'][0][0],     # Translated sentence string
            json_response['input_tokens'][0],       # Tokenized input sentence
            json_response['output_tokens'][0],      # Tokenized translated sentence
            json_response['attentions'][0][0]       # Attention matrix (first layer or head)
        )

    except requests.HTTPError as http_err:
        # Handle and log HTTP-related errors
        print("HTTP error occurred:", http_err)
        print("Response content:", response.text)
        raise

    except Exception as e:
        # Catch-all for any other runtime exceptions
        print("Other error occurred:", e)
        raise


# Handles the translation UI form. On POST, it calls the translation API and renders the result.
@app.route('/projects/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        # Extract form values from user input
        text = request.form.get('source_text', '').strip()
        target_language = request.form.get('target_language', 'fr')
        decode_type = request.form.get('decode_type', 'beam')
        beam_size = int(request.form.get('beam_size', 5))

        # Ensure that the input text is not empty
        if not text:
            return render_template('translate.html', error="Please enter a sentence.")  

        try:
            # Call backend translation API and unpack the result
            results, input_tokens, output_tokens, attention_matrix = call_translation_api(
                [text],
                target_language=target_language,
                decode_type=decode_type,
                beam_size=beam_size
            )
            
            # Render result page with translations and attention map
            return render_template(
                'translate.html',
                source_text=text,
                translation=results,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attentions=attention_matrix,
                decode_type=decode_type,
                beam_size=beam_size
            )
        except Exception as e:
            # Catch and display any errors encountered during processing
            return render_template('translate.html', error=str(e))

    # Render initial empty form for GET requests
    return render_template('translate.html')


# Handles chest X-ray classification via a CNN model. Accepts either uploaded images or sample selections.
@app.route('/projects/medimg', methods=['GET', 'POST'])
def medimg():
    if request.method == 'POST':
        try:
            # Get optional sample image path and model size from form
            sample_image_url = request.form.get('sample_image_url', '').strip()
            model_size = request.form.get('model_size', '').lower()

            # Validate model size input
            if model_size not in ['small', 'large']:
                raise ValueError("Invalid model size selected.")

            if sample_image_url:
                # Process static sample image
                # Strip potential double "static/static/" and normalize path
                sample_image_url = sample_image_url.replace('static/', '').lstrip('/')
                image_path = os.path.join('static', sample_image_url)

                # Confirm file exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Sample image not found: {sample_image_url}")

                # Read sample image as byte stream
                with open(image_path, 'rb') as f:
                    file_bytes = f.read()

                # Format file for requests.post call
                file = ('sample.jpg', file_bytes, 'image/jpeg')

                # Convert to base64 for display in template
                image_b64 = base64.b64encode(file_bytes).decode('utf-8')
            else:
                # Process uploaded image from user
                file = request.files['image']

                # Read and encode image for preview
                file.seek(0)
                image_b64 = base64.b64encode(file.read()).decode('utf-8')

                # Reset file pointer so requests.post can re-read it
                file.seek(0)

            # Send image and model size to remote EC3 prediction API
            response = requests.post(
                EC3_MEDIMG_API_URL,
                files={'image': file},
                data={'model_size': model_size}
            )
            result = response.json()

            # Extract prediction results
            label = "PNEUMONIA" if result['prediction'] == 1 else "NORMAL"
            confidence = result['confidence']
            heatmap = result['heatmap']  # base64-encoded Grad-CAM image
            inference_time = result['inference_time']  # milliseconds

            # Render the result page with prediction and visuals
            return render_template(
                'medimg.html',
                prediction=label,
                confidence=confidence,
                image_url='data:image/png;base64,' + image_b64,
                heatmap_url='data:image/png;base64,' + heatmap,
                inference_time=inference_time,
                model_size=model_size
            )
        except Exception as e:
            # Handle all errors gracefully
            return render_template('medimg.html', error=str(e))

    # Render the initial form on GET request
    return render_template('medimg.html')


# Route to serve the portfolio page
@app.route('/projects/portfolio') 
def portfolio():
    # Render the portfolio.html template
    return render_template('portfolio.html')


# Helper function to check if the SageMaker endpoint is ready
def is_model_ready():
    now = time.time()
    
    # Only check the endpoint status once per minute to reduce API calls
    if now - endpoint_cache['last_checked'] > 60:
        sm_client = boto3.client('sagemaker')  # Create a SageMaker client
        # Fetch the current status of the endpoint
        status = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)['EndpointStatus']
        # Cache the status and timestamp
        endpoint_cache['status'] = status
        endpoint_cache['last_checked'] = now
    
    # Return True if the model is in service and ready to accept requests
    return endpoint_cache['status'] == 'InService'


# Local development entry point
if __name__ == '__main__':
    # Render sets PORT in the environment; fallback to 10000 for local runs
    port = int(os.environ.get("PORT", 10000))
    # Run Flask dev server (not used in production where Gunicorn is used)
    app.run(host='0.0.0.0', port=port, debug=True)
