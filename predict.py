import pandas as pd
from clean_data import clean_data
import pickle



def predict(filename):
    df = pd.read_json('data/' + filename)
    df = clean_data(df, training=False)
    
    with open('data/model.pkl', 'rb') as infile:
        model = pickle.load(infile)
    
    return model.predict(df.values)
    
if __name__ == '__main__':
    print(predict('test.json'))
        
    