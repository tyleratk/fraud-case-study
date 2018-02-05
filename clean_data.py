import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle

def get_fraud(acct_type):
    '''
    return fraud based on acct_type
    '''
    if acct_type in ['fraudster_event', 'fraudster', 'fraudster_att']:
        return 1
    else:
        return 0
        
        

def clean_data(df, training=False):
    '''
    make dataframe using only the rows we want
    '''
    if training:
        keep = ['acct_type', 'body_length', 'name_length', 'num_order', 'num_payouts',
                'previous_payouts', 'user_age', 'user_type', 'gts',
                'sale_duration2'] #'acct_type', 'payout_type',
        
        df = df[keep].copy()
        df['fraud'] = df.acct_type.apply(get_fraud)        
        df.previous_payouts = df.previous_payouts.apply(len)
        df.pop('acct_type')
    else:
        keep = ['body_length', 'name_length', 'num_order', 'num_payouts',
                'previous_payouts', 'user_age', 'user_type', 'gts',
                'sale_duration2'] #'acct_type', 'payout_type',
        df = df[keep].copy()
        df.previous_payouts = df.previous_payouts.apply(len)    
    return df

    
    
def get_training_data(X, y):
    '''
    balances classes for training
    '''
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_sample(X, y)

    return X_resampled, y_resampled
    
    


if __name__ == '__main__':
    pass
    # df = pd.read_json('data/data.json')
    # clean_df = clean_data(df)

    
    
    
    
    
    