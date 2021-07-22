from flask import *
from flask_cors import CORS
import pandas as pd
import joblib

app=Flask(__name__)
CORS(app)
app.config['Debug']=True

@app.route("/predict",methods=['POST','GET'])
def predict_cluster():
    try:
        if d