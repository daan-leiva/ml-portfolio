from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    # Dummy prediction logic
    data = request.get_json()
    prediction = {"result": "This is a dummy prediction."}
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)