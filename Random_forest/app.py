from flask import *
from flask_cors import CORS
import pandas as pd
import joblib

app=Flask(__name__)
CORS(app)
app.config['DEBUG']=True

@app.route("/predict",methods=['POST','GET'])
def forest_predict():
    try:
        if request.json['data'] is not None:
            data=[request.json['data']]
            forest_model=joblib.load('Random_forest.sav')
            df=pd.DataFrame(data)
            print(df)
            y_pred=forest_model.predict(df)
            print(y_pred)
            return Response('The predicted house price is: {}'.format(y_pred))
    except Exception as e:
        return Response(e)

if __name__=="__main__":
    # app.run(host='127.0.0.1',port=5000,debug=True)
    app.run(debug=True)
            