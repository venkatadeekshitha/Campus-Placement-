from flask import Flask, render_template

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return '<h1>Campus Placement App is Running Successfully!</h1>'

# You can add more routes if needed
@app.route('/about')
def about():
    return '<p>This is the about page.</p>'

# For local testing only — not needed in Render
if __name__ == '__main__':
    app.run(debug=True)
