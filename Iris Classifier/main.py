from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('iris_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

@app.route('/')
def index():
    return render_template('index.html', prediction=None)  

@app.route('/submit-data', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

        prediction = loaded_model.predict(input_data)

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
