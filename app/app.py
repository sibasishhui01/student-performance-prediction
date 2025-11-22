from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# load model - path assumes you run this from inside the app/ folder
model_path = os.path.join('..', 'model', 'best_student_model.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # collect form data
    gender = request.form['gender']
    race = request.form['race']
    parent_edu = request.form['parental_education']
    lunch = request.form['lunch']
    test_prep = request.form['test_preparation']
    math = int(request.form['math'])
    reading = int(request.form['reading'])
    writing = int(request.form['writing'])

    # build DataFrame matching training columns
    input_data = pd.DataFrame([{
        'gender': gender,
        'race_ethnicity': race,
        'parental_level_of_education': parent_edu,
        'lunch': lunch,
        'test_preparation_course': test_prep,
        'math score': math,
        'reading score': reading,
        'writing score': writing
    }])

    # predict
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Performance: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
