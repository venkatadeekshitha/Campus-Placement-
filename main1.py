from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return '<h2>🎓 Campus Placement Predictor is running successfully on Render!</h2>'

if __name__ == '__main__':
    app.run(debug=True)
