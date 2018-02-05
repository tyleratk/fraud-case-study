import pandas as pd
from clean_data import clean_data
import pickle



def predict(new_data):
    df = pd.DataFrame([new_data])
    # df = df.from_dict([new_data])
    df = clean_data(df, training=False)
    
    with open('data/model.pkl', 'rb') as infile:
        model = pickle.load(infile)
    
    return model.predict(df.values)[0]
    
if __name__ == '__main__':
    # print(predict('test.json'))
    pass
        
    