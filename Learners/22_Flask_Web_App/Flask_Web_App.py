# Flask_Web_App.py
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/greet', methods=['POST'])
def greet():
    name = request.form.get('name')
    if not name:
        return render_template('index.html', message="Please enter your name.")
    greeting = f"Hello, {name}! Welcome to Flask Web App."
    return render_template('index.html', message=greeting)

if __name__ == '__main__':
    app.run(debug=True)
