from flask import Flask, render_template, request
from src.predict import predict_news
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']

    # Simulate model processing delay for animation
    time.sleep(1.5)
    result = predict_news(title, text)

    color = 'green' if result == 'REAL' else 'red'
    return render_template('result.html', title=title, text=text, prediction=result, color=color)

if __name__ == '__main__':
    app.run(debug=True)
