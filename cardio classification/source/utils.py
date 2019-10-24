import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np

def get_cardio_data(path=os.path.join('data', 'cardio.csv'), test_size=0.3):
    df = pd.read_csv(path, index_col='id', sep=';')
    df['p'] = df['cardio'].apply(lambda x: np.random.rand())
    df = df[((df['cardio'] == 1) & (df['p'] < 0.5)) | (df['cardio'] == 0)]
    df.drop('p', axis=1, inplace=True)
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(df.drop('cardio', axis=1), 
                                                      df['cardio'], 
                                                      test_size=0.25, 
                                                      random_state=17)

    
    for c in ['ap_hi', 'ap_lo', 'weight', 'height']:
        q_h = X_valid[c].quantile(0.985)
        q_l = X_valid[c].quantile(0.015)
        
        X_valid = X_valid[(X_valid[c] < q_h) & (X_valid[c] > q_l)]
    y_valid = X_valid.join(y_valid)['cardio']
    
    return X_train, X_valid, y_train, y_valid
    

