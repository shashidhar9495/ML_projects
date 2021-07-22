from flask import *
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

app=Flask(__name__)
CORS(app)
app.config['DEBUG']=True

@app.route("/predict",methods=['POST','GET'])
def survive_predict():
    try:
        if request.json['data'] is not None:
            data=[request.json['data']]
            tree_model=joblib.load('Decision_Tree.sav')
            column=joblib.load('Columns.sav')
            df=pd.DataFrame(data,columns=column)
            df['Gender']=np.where(df.Sex=='male',1,0)
            df["Parch_enc"]=np.where(df.Parch>=1,1,0)
            df['SibSp_enc']=df['SibSp'].map({0:0,1:1,2:2,3:2,4:2,5:2,8:2})
            df = df.drop(['Sex', 'SibSp', 'Parch'], axis=1)
            print(df)
            prediction=tree_model.predict(df)
            print('Prediction is {}'.format(prediction))
            if prediction[0]==0:
                return Response('Passenger did not survive!!')
            else:
                return Response('Passenger Survived')
    except Exception as e:
        print(e)
        return Response(e)

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)
