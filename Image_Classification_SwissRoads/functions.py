import pandas as pd
import numpy as np
import csv

def load_data(filename, set_name):
    
    '''
    Function to load data saved as npz files
    '''
    

    # Load data
    with np.load(filename, allow_pickle=False) as npz_file:

        # Load items into a dictionary
        #data_dict = dict(npz_file.items())
        
        X = npz_file['data']
        y = npz_file['labels']
        y_labels = npz_file['names']
        features = npz_file['features']
        filenames = npz_file['filename'] 
        
        str_info = set_name +' data info:'
        print(str_info)
        print('-'*len(str_info))
        print('X:', X.shape)
        print('y:', y.shape)
        print('labels:', y_labels.shape)
        print('features:', features.shape)
        print('filenames', filenames.shape, '\n')

    return X, y, y_labels, features, filenames
    

def merge_tr_val_sets():
    
    '''
    Function to merge training and validation data
    '''
    
    X_tr, y_tr, y_tr_labels, train_features, train_filenames = load_data('trainfile_mobile_v2.npz', 'Training')
    X_val, y_val, y_val_labels, val_features, val_filenames = load_data('validfile_mobile_v2.npz', 'Validation')
    
    X_tr_merged = np.concatenate((X_tr, X_val), axis=0)
    y_tr_merged = np.concatenate((y_tr, y_val), axis=0)
    y_tr_labels_merged = y_tr_labels #np.concatenate((y_tr_labels, y_val_labels), axis=1)
    train_features_merged = np.concatenate((train_features, val_features), axis=0)
    train_filenames_merged = np.concatenate((train_filenames, val_filenames), axis=0)
    
    str_info = 'merged training and validation data info:'
    print(str_info)
    print('-'*len(str_info))
    print('X:', X_tr_merged.shape)
    print('y:', y_tr_merged.shape)
    print('labels:', y_tr_labels_merged.shape)
    print('features:', train_features_merged.shape)
    print('filenames', train_filenames_merged.shape, '\n')    
    
    return X_tr_merged, y_tr_merged, y_tr_labels_merged, train_features_merged, train_filenames_merged



def save_test_accuracy(filename, model, acc):
    
    '''
    Function to save test accuracy
    '''
    
    d = {'model': model, 'test_accuracy': acc}
    
    with open(filename, 'w') as f: 
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)
        
        


def get_all_results():
    
    '''
    Function to read all csv files containing the test accuracy from the different tests.
    All results are merged in a dataframe.
    '''
    
    df_knn = pd.read_csv('knn.csv')
    df_dt = pd.read_csv('decision_tree.csv')
    df_log = pd.read_csv('logistic.csv')
    df_rf = pd.read_csv('random_forest.csv')
    df_svm_linear = pd.read_csv('svm_linear.csv')
    df_svm_rbf = pd.read_csv('svm_rbf.csv')
    df_1layer_nn = pd.read_csv('1_layer_nn.csv')
    df_2layer_nn = pd.read_csv('2_layer_nn.csv')
    df_cnn = pd.read_csv('cnn.csv')
    
    df_merged = pd.concat([df_knn, df_dt, df_log, df_rf, df_svm_linear, df_svm_rbf, df_1layer_nn, df_2layer_nn, df_cnn], 
                          ignore_index=True, sort=False)
    
    return df_merged