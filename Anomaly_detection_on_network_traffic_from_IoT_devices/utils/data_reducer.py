import os
import re
import pandas as pd
import numpy as np

# Create new dataset by removing unnecessary entries
def drop_entries(df):
    
    # Get a copy of the dataframe
    df = df.copy()
    
    # Replace encoded missing values with nans
    df = df.replace({'-':np.nan})
    df['tunnel_parents'] = df['tunnel_parents'].replace({np.nan:'-'})
    df['detailed_label'] = df['detailed_label'].replace({np.nan:'-'})
#     df['origin_address'] = df['origin_address'].replace({'::':np.nan})
#     df['response_address'] = df['response_address'].replace({'::':np.nan})    
    
    
    # Drop IPv6 addresses
    mask = df.loc[df['response_address'].str.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")!= True].index

    df = df.drop(df.index[[mask]], axis=0)
    df = df.reset_index(drop=True)  
    
    # Drop low entries classes
    classes_to_drop = ['C&C-Torii', 'C&C-FileDownload', 'FileDownload']

    for label in classes_to_drop:
        mask = df.loc[df['detailed_label']==label].index
        df = df.drop(df.index[[mask]], axis=0)
        df = df.reset_index(drop=True)   
        
    # Drop outliers    
    cols_change_dtype_ = ['duration', 'orig_bytes', 'orig_ip_bytes', 'resp_ip_bytes', 'missed_bytes']    
    
    # Change data type 
    for c in cols_change_dtype_:        
        df[c] = df[c].astype('float')          
    
    mask = df.loc[(df['duration']>500) | 
                  (df['orig_bytes']>=1e5) | 
                  (df['orig_ip_bytes']>=1e4) | 
                  (df['resp_ip_bytes']>=1e4) |
                  (df['missed_bytes']>1)].index

    df = df.drop(df.index[[mask]], axis=0)
    df = df.reset_index(drop=True)           
               
    # Save to file
    df.to_csv(os.path.join('data', 'ioT_data_reduced.csv'), index=False)  
    
    return df