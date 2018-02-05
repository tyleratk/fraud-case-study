import pandas as pd 
import requests
import json
import sched, time
from pymongo import MongoClient
import pickle
from clean_data import clean_data


def request(url,table):
    '''
    input: 
        - url to make requests from 
    output: 
        - data point as a json file 
    '''
    data_point = requests.get(url).json()
    table.insert_one(data_point)
    object_id = data_point['object_id']
    return data_point, object_id
    
def predict(new_data, model):
    df = pd.DataFrame([new_data])
    df = clean_data(df, training=False)
    predition = model.predict(df.values)[0]
    proba = model.predict_proba(df.values)[0][1]
    return prediction, proba

if __name__=='__main__':    
    URL = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'

    '''
    - instantiate mongo database 
    - define the database 
    - create fraud table in the database 
    '''
    
    db_client = MongoClient()
    db = db_client['fraud_case_study']
    table = db['fraud']
    
    new_data, object_id = request(URL,table)  
    
    prediction = predict(new_data)
    
    table.update_one({'object_id':object_id},{'$set':{'prediction':int(prediction)}})
    
    
    
    