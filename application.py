from flask import Flask,jsonify,request,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

#importing ridge and scaler 
ridge_model=pickle.load(open('models/Ridgecv.pkl','rb'))
scaler_standard=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def prediction_closing():
    if request.method=='POST':
        Oprice=float(request.form.get('Opening price'))
        new_data_scaled=scaler_standard.transform([[Oprice]])
        results=ridge_model.predict(new_data_scaled)
                                   
        return render_template('home.html',result = results[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
    