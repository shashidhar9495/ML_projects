from flask import *
from flask_cors import CORS
import pandas as pd
import joblib

app=Flask(__name__)
CORS(app)
app.config['DEBUG']=True

@app.route("/predict",methods=['POST','GET'])
def predict_val():
    try:
        if request.json['data'] is not None:
            data=[request.json['data']]
            boost_model=joblib.load('XGB.sav')
            dictionary=joblib.load('dict.sav')
            column=joblib.load('col.sav')
            df=pd.DataFrame(data,columns=column)
            df['Country']=df.native_country.map(dictionary)
            print(df)
            y_pred=boost_model.predict(df)
            print('Predicted val is {}'.format(y_pred))
            if y_pred == 0:
                return Response('Belongs to wage class <= 50k')
            else:
                return Response('Belongs to wage class >50k')
    except Exception as e:
        print(e)
        return Response(e)
if __name__=="__main__":
    # app.run(host='127.0.0.1',port=5000,debug=True)
    app.run(debug=True)