# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:58:21 2021

@author: umutc
"""

from flask import Flask, request, render_template
#import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#ML Part
#df = pd.read_csv('heart.csv')
#y = df["target"]
#X = df.drop('target',axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
#input_vitals = X_train.iloc[0]
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
#model_knn = 'K-NeighborsClassifier'
#knn = KNeighborsClassifier(n_neighbors=10)
#knn.fit(X_train, y_train)
#knn_predicted = knn.predict(X_test)
#knn_conf_matrix = confusion_matrix(y_test, knn_predicted)
#knn_acc_score = accuracy_score(y_test, knn_predicted)
#print("confusion matrix")
#print(knn_conf_matrix)
#print("-------------------------------------------")
#print("Accuracy of K-NeighborsClassifier:",knn_acc_score*100,'\n')
#print("-------------------------------------------")
#print(classification_report(y_test,knn_predicted))

knn = joblib.load('knnmodel.pkl' , mmap_mode ='r')

#Flask
app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True

#App pages
@app.route("/")
def welcomepage():
    return render_template("welcome.html")

@app.route("/start")
def startpage():
    return render_template("start.html")


@app.route("/personalinfo", methods=['GET', 'POST'])
def personalinfopage():
    return render_template("personalinfo.html")

@app.route("/results", methods=['GET', 'POST'])
def resultspage():
    #Getting the inputs
    myinput = {'age': request.args.get('age', 40), 'sex': request.args.get('selectsex', 0), 'cp':request.args.get('selectchest', 0) , 
               'trestbps':request.args.get('rbp', 135), 'chol':request.args.get('secho', 250), 'fbs':request.args.get('fbs', 0), 
               'restecg':request.args.get('ecg', 0), 'thalach':request.args.get('heartrate', 160), 'exang': request.args.get('angina', 0),
               'oldpeak': request.args.get('stseg', 0), 'slope':request.args.get('slope', 0), 'ca':request.args.get('vessel', 0),
               'thal':request.args.get('thalassemia', 2)}
    
    #input_vitals[0] = request.args.get('age', 40)
    #input_vitals[1] = request.args.get('selectsex', 0)
    #input_vitals[2] = request.args.get('selectchest', 0)
    #input_vitals[3] = request.args.get('rbp', 135)
    #input_vitals[4] = request.args.get('secho', 250)
    #input_vitals[5] = request.args.get('fbs', 0)
    #input_vitals[6] = request.args.get('ecg', 0)
    #input_vitals[7] = request.args.get('heartrate', 160)
    #input_vitals[8] = request.args.get('angina', 0)
    #input_vitals[9] = request.args.get('stseg', 0)
    #input_vitals[10] = request.args.get('slope', 0)
    #input_vitals[11] = request.args.get('vessel', 0)
    #input_vitals[12] = request.args.get('thalassemia', 2)
    #THE MODEL WILL DO THE PREDICTION HERE
    prediction = knn.predict(np.array(list(myinput.values())).astype(float).reshape(1, -1))
    print(prediction)
    if prediction[0] == 0:
        return render_template("good.html")
    else:
        return render_template("warning.html")

    
if __name__ == "__main__":
    app.run(debug=True)