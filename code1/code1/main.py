import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/add', methods=['POST'])
def add():
    model = joblib.load("mymodel.h5")
    feature_names = model.feature_names_in_
    
    a = int(request.form['age'])
    b = str(request.form['gender'])
    c = str(request.form['stream'])
    d = int(request.form['internship'])
    e = int(request.form['cgpa'])
    f = int(request.form['hostel'])
    g = int(request.form['backlogs'])
    
    new_data = {
        "Age": a,
        "Gender": b,
        "Stream": c,
        "Internships": d,
        "CGPA": e,
        "Hostel": f,
        "History of backlogs": g,
    }
    new_data_df = pd.DataFrame([new_data])
    new_data_df = pd.get_dummies(new_data_df)
    new_data_encoded = new_data_df.reindex(columns=feature_names, fill_value=0)
    result = model.predict(new_data_encoded)

    return redirect(url_for('result', result=result[0]))

@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
