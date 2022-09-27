from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
import ipaddress
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

#################################################################
## Recover Missing values 
#################################################################    
class RecoverNansPreprocessor(BaseEstimator, TransformerMixin):
    '''
    Recover missing values
    '''

    def __init__(self):
        self.columns = []
        self.cols_change_dtype_ = ['duration', 
                                   'orig_bytes', 
                                   'resp_bytes', 
                                   'origin_port', 
                                   'response_port', 
                                   'missed_bytes', 
                                   'orig_pkts',
                                   'orig_ip_bytes',
                                   'resp_pkts',
                                   'resp_ip_bytes'
                                  ]              
        
    # Function to replace nulls in continuous variables    
    def preprocess_f(self, X_df):
        # Work on a copy
        X_df = X_df.copy()
        
        # Replace encoded missing values with nans
        X_df = X_df.replace({'-':np.nan})
        X_df['origin_address'] = X_df['origin_address'].replace({'::':np.nan})
        X_df['response_address'] = X_df['response_address'].replace({'::':np.nan})     
        
        # Change data type 
        for c in self.cols_change_dtype_:
            X_df[c] = X_df[c].astype('float')  

        return X_df
        

    def fit(self, X_df, y=None):

        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Save columns names/order for inference time
        self.columns_ = X_preprocessed.columns

        return self  

    def transform(self, X_df):

        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Make sure to have the same features
        X_reindexed = X_preprocessed.reindex(columns=self.columns_, fill_value=0)
        
        # Update columns
        self.columns_ = X_reindexed.columns

        return X_reindexed
    
    # Needed to get the name of the features out to compare the coefficients
    def get_features_name(self):        
        return self.columns    

    
#################################################################
## Cleaning 
#################################################################    
class CleaningPreprocessor(BaseEstimator, TransformerMixin):
    '''
    Drop columns and fill nulls
    '''
    
    def __init__(self):
        self.columns = []
        self.cols_to_drop_ = ['local_orig', 'local_resp', 'tunnel_parents', 'service', 'uid', 'timestamp', 'missed_bytes']
        self.cols_for_mean_ = ['duration']        
        
    # Function to replace nulls in continuous variables    
    def preprocess_f(self, X_df, train_mean):
        # Work on a copy
        X_df = X_df.copy()

        # Drop columns
        X_df = X_df.drop(self.cols_to_drop_, axis=1)
                
        # Missing values to be filled with the mean                
        for c in self.cols_for_mean_:
            X_df[c] = X_df[c].fillna(train_mean[c])   
                
        # Other missing values
        X_df['orig_bytes'] = X_df['orig_bytes'].fillna(0) 
        X_df['resp_bytes'] = X_df['resp_bytes'].fillna(X_df['resp_ip_bytes']) 
        X_df['history'] = X_df['history'].fillna('no_history')   
         

        return X_df
        

    def fit(self, X_df, y=None):

        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Save train mean for continuous variables
        self.train_mean = X_df[self.cols_for_mean_].mean()

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df, self.train_mean)

        # Save columns names/order for inference time
        self.columns_ = X_preprocessed.columns

        return self  

    def transform(self, X_df):

        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df, self.train_mean)

        # Make sure to have the same features
        X_reindexed = X_preprocessed.reindex(columns=self.columns_, fill_value=0)
        
        # Update columns
        self.columns_ = X_reindexed.columns

        return X_reindexed
    
    # Needed to get the name of the features out to compare the coefficients
    def get_features_name(self):        
        return self.columns    
    


#################################################################
## Categorical Preprocessor 
#################################################################    
class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    
    '''
    Convert categorical data to dummies and/or numeric. 
    '''
    
    def __init__(self, cols_to_dummies=[], cols_to_numeric=[] ):
        self.columns = []
        self.cols_to_dummies = cols_to_dummies
        self.cols_to_numeric = cols_to_numeric
                    
    def preprocess_f(self, X_df):
        
        # Work on a copy
        X_df = X_df.copy()        
        
        # One-hot encoding 
        if len(self.cols_to_dummies)!=0:
            X_df = pd.get_dummies(X_df, columns=self.cols_to_dummies, dtype=np.float64)  
        
        # Numeric Representation
        if len(self.cols_to_numeric)!=0:
            for c in X_df[self.cols_to_numeric]:
                X_df[c] = X_df[c].astype('category')
                X_df[c] = X_df[c].cat.codes   
                X_df[c] = X_df[c].astype('float64')
        
            
        return X_df

    def fit(self, X_df, y=None):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Save columns names/order for inference time
        self.columns = X_preprocessed.columns

        return self

    def transform(self, X_df):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Make sure to have the same features
        X_reindexed = X_preprocessed.reindex(columns=self.columns, fill_value=0)
        
        self.columns = X_reindexed.columns

        return X_reindexed
    
    def get_features_name(self):
        return self.columns  

    
#################################################################
## Numerical Preprocessor 
#################################################################    
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    
    '''
    This preprocessor allows to add or replace the numerical features
    with log, power or quantile transformations
    '''
    
    def __init__(self, cols_to_logs=[], cols_to_pt=[], cols_to_quantile=[], replace=False):
        self.columns = []
        self.cols_to_logs = cols_to_logs
        self.cols_to_pt = cols_to_pt
        self.cols_to_quantile = cols_to_quantile
        self.replace = replace
                    
    def preprocess_f(self, X_df):
        
        if len(self.cols_to_pt)!=0:
            pt = PowerTransformer(standardize=True)   
            
        if len(self.cols_to_quantile)!=0:   
            qt = QuantileTransformer(output_distribution='normal', random_state=0, copy=False)
        
        if self.replace == False:
        
            # Add log transformations
            for c in self.cols_to_logs:
                X_df[c+'_log'] = X_df[c].apply(lambda x: np.log1p(x))

            # Add power transformations
            for c in self.cols_to_pt:
                X_df[c+'_pt'] = pt.fit_transform(X_df[[c]])
                
            # Add Quantile transformations
            for c in self.cols_to_quantile:
                X_df[c+'_qt'] = qt.fit_transform(X_df[[c]])   

                
        else: # Replace initial features with transformations
            
            for c in self.cols_to_logs:
                X_df[c] = X_df[c].apply(lambda x: np.log1p(x))

            for c in self.cols_to_pt:
                X_df[c] = pt.fit_transform(X_df[[c]])
                
            for c in self.cols_to_quantile:
                X_df[c] = qt.fit_transform(X_df[[c]])                 
            
        return X_df
    

    def fit(self, X_df, y=None):
        
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Save columns names/order for inference time
        self.columns = X_preprocessed.columns

        return self
    

    def transform(self, X_df):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Make sure to have the same features
        X_reindexed = X_preprocessed.reindex(columns=self.columns, fill_value=0)
        
        self.columns = X_reindexed.columns

        return X_reindexed
    
    def get_features_name(self):
        return self.columns  
    
    
#################################################################
## IP Encoding Preprocessor 
#################################################################    
class IPEncodingPreprocessor(BaseEstimator, TransformerMixin):
    
    '''
    Encode IP addresses either to 8bit octets or to integers by setting
    the flag ip_to_octets equals to True or to False respectively. 
    
    Note: For testing the encoding to integers should be the 1st trial
    because the encoding to octets call will drop the original features
    origin_address and response_address. 
    '''
    
    def __init__(self, ip_to_octets=False, ip_to_integers=False):
        self.columns = []
        self.ip_to_octets = ip_to_octets
        self.ip_to_integers = ip_to_integers
        
    def encoding_to_octets(self, X_df, col, prefix):
        
        # Split IP address
        X_df[col] = X_df[col].apply(lambda x: x.split('.'))
        
        # Gets octades
        X_df[prefix + '_oct1'] = X_df[col].apply(lambda x: x[0]).astype(np.float64)
        X_df[prefix + '_oct2'] = X_df[col].apply(lambda x: x[1]).astype(np.float64)
        X_df[prefix + '_oct3'] = X_df[col].apply(lambda x: x[2]).astype(np.float64)
        X_df[prefix + '_oct4'] = X_df[col].apply(lambda x: x[3]).astype(np.float64)
                
        return X_df
            
                    
    def preprocess_f(self, X_df):
        
        # Work on a copy
        X_df = X_df.copy()
        
        if (self.ip_to_integers==True):
            
            # Encode to integers
            X_df['origin_address_int'] = X_df.apply(lambda row: int(ipaddress.IPv4Address(row.origin_address)), axis = 1) 
            X_df['response_address_int'] = X_df.apply(lambda row: int(ipaddress.IPv4Address(row.response_address)), axis = 1)         
        
        if (self.ip_to_octets==True): 
        
            # Separate orig_address and resp_address
            X_df = self.encoding_to_octets(X_df, 'origin_address', 'orig_add')
            X_df = self.encoding_to_octets(X_df, 'response_address', 'resp_add')
            
        X_df = X_df.drop(['origin_address', 'response_address'], axis=1)
                        
        return X_df

    def fit(self, X_df, y=None):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Save columns names/order for inference time
        self.columns = X_preprocessed.columns

        return self

    def transform(self, X_df):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Make sure to have the same features
        X_reindexed = X_preprocessed.reindex(columns=self.columns, fill_value=0)
        
        self.columns = X_reindexed.columns

        return X_reindexed
    
    def get_features_name(self):
        return self.columns  
    
#################################################################
## Add Binaries Preprocessor 
#################################################################    
class AddBinariesPreprocessor(BaseEstimator, TransformerMixin):
    
    '''
    Binary features are added coming from missing bytes and/or ipaddress lib
    '''
    
    def __init__(self, has_ip_address_features=False):
        self.columns = []
        self.has_ip_address_features = has_ip_address_features
                    
    def preprocess_f(self, X_df):
        
        # Work on a copy
        X_df = X_df.copy()        
        
        if self.has_ip_address_features==True:
            X_df['origin_is_private'] = X_df.apply(lambda row: int(ipaddress.ip_address(row.origin_address).is_private), axis = 1)
            X_df['origin_is_global'] = X_df.apply(lambda row: int(ipaddress.ip_address(row.origin_address).is_global), axis = 1)
            X_df['response_is_private'] = X_df.apply(lambda row: int(ipaddress.ip_address(row.response_address).is_private), axis = 1)
            X_df['response_is_global'] = X_df.apply(lambda row: int(ipaddress.ip_address(row.response_address).is_global), axis = 1)
 
        return X_df

    def fit(self, X_df, y=None):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Save columns names/order for inference time
        self.columns = X_preprocessed.columns

        return self

    def transform(self, X_df):
        # Check that we get a DataFrame
        assert type(X_df) == pd.DataFrame

        # Preprocess data
        X_preprocessed = self.preprocess_f(X_df)

        # Make sure to have the same features
        X_reindexed = X_preprocessed.reindex(columns=self.columns, fill_value=0)
        
        self.columns = X_reindexed.columns

        return X_reindexed
    
    def get_features_name(self):
        return self.columns      
    
 