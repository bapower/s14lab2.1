from flask import Flask, render_template
import joblib
import logging

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

app = Flask(__name__)


@app.route('/')
def index():
    linearRegression = joblib.load('./models/linearRegression.pkl')
    decisionTree = joblib.load('./models/decisionTree.pkl')
    # Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
    linearRegressionPrediction = str(linearRegression.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1))
    decisionTreePrediction = str(decisionTree.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0])
    return (render_template('index.html', 
    	linearRegressionPrediction = linearRegressionPrediction, 
    	decisionTreePrediction = decisionTreePrediction))