from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas 
import sklearn

#importing model
# model=pickle.load(open('model.pkl','rb'))
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
#creating a flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humid = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rain = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humid, ph, rain]
    features = np.array(feature_list).reshape(1, -1)
    prediction = model.predict(features)
    
    crop_dict={20: 'rice', 11: 'maize', 3: 'chickpea', 9: 'kidneybeans', 18: 'pigeonpeas', 13: 'mothbeans', 14: 'mungbean', 2: 'blackgram', 10: 'lentil', 19: 'pomegranate', 1: 'banana', 12: 'mango', 7: 'grapes', 21: 'watermelon', 15: 'muskmelon', 0: 'apple', 16: 'orange', 17: 'papaya', 4: 'coconut', 6: 'cotton', 8: 'jute', 5: 'coffee'}
    
    if prediction[0] in crop_dict:
        predicted_crop = crop_dict[prediction[0]]
        result = "{} is the best crop for the cultivation".format(predicted_crop)
    else:
        result = "Please Check all the values or else Sorry we are not able to recommend the crop"
    return render_template('index.html',result = result)

# python main
if __name__ == "__main__":
    app.run(debug=True)