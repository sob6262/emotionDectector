from flask import Flask, render_template, request
from transformers import pipeline

# Initialize the Flask application
app = Flask(__name__)

# Initialize transformer model for text-based emotion analysis
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Route for main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for text input for emotion analysis
@app.route('/analyze', methods=['POST'])
def analyze_text():
    input_text = request.form['input']
    
    # Get emotion prediction from transformer model
    result = emotion_model(input_text)[0]
    
    emotion = result['label']
    score = result['score']
    
    return render_template('index.html', emotion=emotion, score=score, input_text=input_text)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
