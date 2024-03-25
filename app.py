import pickle
from flask import Flask, request,app, jsonify,url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## loading the pickle file

regmodel = pickle.load(open('reg.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data'] # we are getting the data in json format.
    print(data)
    data = np.array(list(data.values())).reshape(1,-1)
    output = regmodel.predict(data)
    print(output[0])   ## output is a 2d array.
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Gold price Predicted to be {}".format(output))



if __name__ == "__main__":
    app.run(debug=True)

