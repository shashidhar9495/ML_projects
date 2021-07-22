from flask import *
from flask_cors import CORS
import pandas as pd
import joblib

app=Flask(__name__)
CORS(app)
app.config['DEBUG']=True

@app.route("/predict",methods=['POST','GET'])


def predict_rating():
    try:
        if request.json['data'] is not None:
            data=[request.json['data']]
            model=joblib.load('reg_model.sav')
            columns=joblib.load('Columns.sav')
            df=pd.DataFrame(data,columns=columns)
            df['attacking_work_rate'] = df.attacking_work_rate.apply(series_convert)
            df['defensive_work_rate'] = df.defensive_work_rate.apply(series_convert)
            print(df)
            y_pred=model.predict(df)
            print(y_pred)
            return Response('The Predicted Player Rating is :{}'.format(float(y_pred)))
    except Exception as e:
        return Response(e)
def series_convert(series):
    if series == 'medium':
        return 'medium'
    elif series == 'high':
        return 'high'
    elif series== 'low':
        return 'low'
    else:
        return 'others'

if __name__=="__main__":
    # app.run(host='127.0.0.1',port=5000,debug=True)
    app.run(debug=True)