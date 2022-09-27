import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.base import clone
import sklearn.metrics as metrics 

from itertools import compress


###################################
# Function to be used for testing 
###################################
def fit_pipe(X, y, pipe, verbose=False, sample_weight=None ):
    
    gs_results = []    
    
    # Split train/test sets into different sizes
    for test_size in np.arange(0.2, 0.6, 0.1):
        

        # For each train/set size run 10 different random splittings
        for run_idx in range(10):
            

            # Split sets        
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=run_idx, stratify=y)  

            if verbose==True:
                pipe.fit(X_tr, y_tr, verbose=True)
            else:
                pipe.fit(X_tr, y_tr)

            # Evaluate predictions
            y_pred_tr = pipe.predict(X_tr)  
            y_pred_val = pipe.predict(X_val) 
            

            # Store results
            gs_results.append({
                'test_size': test_size,
                'run_idx': run_idx,
                'f1_tr': metrics.f1_score(y_tr, y_pred_tr, average='weighted', sample_weight=sample_weight),
                'f1_val': metrics.f1_score(y_val, y_pred_val, average='weighted', sample_weight=sample_weight),
                'y_tr': y_tr,
                'y_pred_tr': y_pred_tr,                 
                'y_val': y_val,
                'y_pred_val': y_pred_val, 
            })  

    gs_results = pd.DataFrame(gs_results)

    return gs_results



##########################################################
# Average results
##########################################################
def average_results(df, vertical_scans=False):
    
    results = []
    
    for test_size in np.arange(0.2, 0.6, 0.1):
        
        conf_matrix_tr = []
        conf_matrix_val = []      
    
        df_loc = df.loc[df['test_size']==test_size]

        # Average all random states and display results
        results.append({
            'val_size': int(test_size*100),
            'f1_tr_mean': df_loc['f1_tr'].mean(),
            'f1_tr_std': df_loc['f1_tr'].std(),
            'f1_val_mean': df_loc['f1_val'].mean(),
            'f1_val_std': df_loc['f1_val'].std(),           

        })
 

        ############################################### 
        #  Confusion Matrices 
        ################################################
        # Train data
        df_ytr = pd.DataFrame(df_loc['y_tr'].tolist())
        df_ytr = df_ytr.T

        df_pred_ytr = pd.DataFrame(df_loc['y_pred_tr'].tolist())
        df_pred_ytr = df_pred_ytr.T

        for c in df_ytr.columns:
            matrix =  metrics.confusion_matrix(y_true=df_ytr[c], y_pred=df_pred_ytr[c])
            conf_matrix_tr.append(matrix)  


        # Validation data
        df_yval = pd.DataFrame(df_loc['y_val'].tolist())
        df_yval = df_yval.T

        df_pred_yval = pd.DataFrame(df_loc['y_pred_val'].tolist())
        df_pred_yval = df_pred_yval.T

        for c in df_yval.columns:
            matrix =  metrics.confusion_matrix(y_true=df_yval[c], y_pred=df_pred_yval[c])
            conf_matrix_val.append(matrix)    
            

        ##################################
        # Get average confusion matrices
        ##################################
        if vertical_scans==False:
            labels = ['Benign', 'HPortScan', 'C&C', 'DDoS', 'Attack']
        else:
            labels = ['Benign', 'HPortScan', 'C&C', 'DDoS', 'Attack', 'VPortScan']

        # Average confusion matrices and normalize them
        conf_matrix_tr_mean = np.average(conf_matrix_tr, axis=0)
        conf_matrix_tr_mean_normalized = conf_matrix_tr_mean/ conf_matrix_tr_mean.sum(axis=1)[:, np.newaxis]

        conf_matrix_val_mean = np.average(conf_matrix_val, axis=0)
        conf_matrix_val_mean_normalized = conf_matrix_val_mean/ conf_matrix_val_mean.sum(axis=1)[:, np.newaxis]     

        fig, ax = plt.subplots(1,2, figsize=(10,4))

        sns.heatmap(conf_matrix_tr_mean_normalized, annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels, ax=ax[0])
        ax[0].set_ylabel('Actual')
        ax[0].set_xlabel('Predicted')
        ax[0].set_title('Training data, size: {:.0f}%'.format((1-test_size)*100))

        sns.heatmap(conf_matrix_val_mean_normalized, annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels, ax=ax[1])
        ax[1].set_ylabel('Actual')
        ax[1].set_xlabel('Predicted')  
        ax[1].set_title('Validation data, size: {:.0f}%'.format((test_size)*100))
     


        plt.tight_layout()
        plt.show(block=False)    

    results = pd.DataFrame(results)
    display(results)
        
    return results

##########################################################
# Plot the PVE from PCA
##########################################################

def plot_pve(pve, pipe):
    
    plt.figure(figsize=(20, 5))

    # Plot pve with cumulative sum
    xcor = np.arange(1, len(pve) + 1) # 1,2,..,n_components
    plt.bar(xcor, pve)
    plt.xticks(xcor)

    # Add cumulative sum
    pve_cumsum = np.cumsum(pve)
    plt.step(
        xcor+0.5, # 1.5,2.5,..,n_components+0.5
        pve_cumsum, # Cumulative sum
        label='cumulative'
    )

    # Add labels
    plt.xlabel('principal component', fontsize=18)
    plt.ylabel('proportion of variance explained', fontsize=18)
    plt.xticks(np.arange(1, pipe.named_steps['ft'].n_components_, 5), rotation=45)
    plt.yticks(np.arange(0.1, 1.1, 0.1))
    plt.title('Scree plot with cumulative PVE', fontsize=15)
    plt.legend()
    plt.grid(linewidth = 2)
    plt.show()    
    
    # Plot pve alone w/o the cumulative sum
    plt.plot(pve)
    plt.xlabel('principal component', fontsize=13)
    plt.ylabel('proportion of variant explained', fontsize=13)
    plt.title('Scree plot', fontsize=15)
    plt.grid()
    plt.show()

##########################################################
# Plot 2 first principal components
##########################################################    
def plot_2_pp(X,y, pipe):
    
    labels = ['Benign', 'PartOfAHorizontalPortScan', 'C&C', 'DDoS', 'Attack', 'PartOfAVerticalPortScan']
    
    
    # transform train features
    X_transf = pipe.transform(X)

    fig = plt.figure(figsize=(7, 5))    

    # Plot each category
    for cat in np.unique(y).tolist():

        idx = (y == cat)    

        # Plot their first two pca components
        plt.scatter(
            X_transf[idx, 0], X_transf[idx, 1],
            label=labels[cat],
            alpha=0.5
        )

    # Labels and legend    
    plt.xlabel('1st component', fontsize=12)
    plt.ylabel('2nd component', fontsize=12)
    plt.legend(bbox_to_anchor=(1.5,1.05), fontsize=12)
    plt.show()

##########################################################
# Plot original data against each principal component
##########################################################      
from sklearn.decomposition import PCA
    
def transform_pca(X, n):

    pca = PCA(n_components=n)
    pca.fit(X)
    X_new = pca.inverse_transform(pca.transform(X))

    return X_new

def plot_orig_data_and_principalComponents(X, pca_pipe):

    # Clone pipe
    scaler = clone(pca_pipe)
    # Remove PCA 
    scaler.steps.pop(-1)

    X_scaled = scaler.fit_transform(X)

    # Define subplots
    rows = 5; cols = 4; comps = 1;
    fig, axes = plt.subplots(rows, cols, figsize=(12,12), sharex=True, sharey=True)

    # Draw
    for row in range(rows):
        for col in range(cols):
            X_new = transform_pca(X_scaled, comps)
            ax = sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], ax=axes[row, col], color='grey', alpha=.3)
            ax = sns.scatterplot(x=X_new[:, 0], y=X_new[:, 1], ax=axes[row, col], color='blue', alpha=.7)
            ax.set_title(f'PCA Components: {comps}')
            comps += 1

    plt.tight_layout()
    plt.show()    
    
##########################################
# Fit a pipeline with single splitting 
############################################
def fit_pipe_single_splitting(X, y, pipe, verbose=False, sample_weight=None ):
    
    gs_results = []    
    
    # Run 10 different random splittings
    for run_idx in range(10):

        # Split sets        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.4, random_state=run_idx, stratify=y)  

        if verbose==True:
            pipe.fit(X_tr, y_tr, verbose=True)
        else:
            pipe.fit(X_tr, y_tr)

        # Evaluate predictions
        y_pred_tr = pipe.predict(X_tr)  
        y_pred_val = pipe.predict(X_val) 


        # Store results
        gs_results.append({
            'run_idx': run_idx,
            'f1_tr': metrics.f1_score(y_tr, y_pred_tr, average='weighted', sample_weight=sample_weight),
            'f1_val': metrics.f1_score(y_val, y_pred_val, average='weighted', sample_weight=sample_weight),
            'y_tr': y_tr,
            'y_pred_tr': y_pred_tr,                 
            'y_val': y_val,
            'y_pred_val': y_pred_val, 
        })  

    gs_results = pd.DataFrame(gs_results)

    return gs_results    

##########################################################
# Average results for single splitting
##########################################################
def average_results_single_splitting(df):
    
    test_size = 0.4
    
    results = []
    conf_matrix_tr = []
    conf_matrix_val = []      


    # Average all random states and display results
    results.append({
        'f1_tr_mean': df['f1_tr'].mean(),
        'f1_tr_std': df['f1_tr'].std(),
        'f1_val_mean': df['f1_val'].mean(),
        'f1_val_std': df['f1_val'].std(),           

    })


    ############################################### 
    #  Confusion Matrices 
    ################################################
    # Train data
    df_ytr = pd.DataFrame(df['y_tr'].tolist())
    df_ytr = df_ytr.T

    df_pred_ytr = pd.DataFrame(df['y_pred_tr'].tolist())
    df_pred_ytr = df_pred_ytr.T

    for c in df_ytr.columns:
        matrix =  metrics.confusion_matrix(y_true=df_ytr[c], y_pred=df_pred_ytr[c])
        conf_matrix_tr.append(matrix)  


    # Validation data
    df_yval = pd.DataFrame(df['y_val'].tolist())
    df_yval = df_yval.T

    df_pred_yval = pd.DataFrame(df['y_pred_val'].tolist())
    df_pred_yval = df_pred_yval.T

    for c in df_yval.columns:
        matrix =  metrics.confusion_matrix(y_true=df_yval[c], y_pred=df_pred_yval[c])
        conf_matrix_val.append(matrix)    


    ##################################
    # Get average confusion matrices
    ##################################
    labels = ['Benign', 'HPortScan', 'C&C', 'DDoS', 'Attack', 'VPortScan']

    # Average confusion matrices and normalize them
    conf_matrix_tr_mean = np.average(conf_matrix_tr, axis=0)
    conf_matrix_tr_mean_normalized = conf_matrix_tr_mean/ conf_matrix_tr_mean.sum(axis=1)[:, np.newaxis]

    conf_matrix_val_mean = np.average(conf_matrix_val, axis=0)
    conf_matrix_val_mean_normalized = conf_matrix_val_mean/ conf_matrix_val_mean.sum(axis=1)[:, np.newaxis]     

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    sns.heatmap(conf_matrix_tr_mean_normalized, annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels, ax=ax[0])
    ax[0].set_ylabel('Actual')
    ax[0].set_xlabel('Predicted')
    ax[0].set_title('Training data, size: {:.0f}%'.format((1-test_size)*100))

    sns.heatmap(conf_matrix_val_mean_normalized, annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels, ax=ax[1])
    ax[1].set_ylabel('Actual')
    ax[1].set_xlabel('Predicted')  
    ax[1].set_title('Validation data, size: {:.0f}%'.format((test_size)*100))


    plt.tight_layout()
    plt.show(block=False)    

    results = pd.DataFrame(results)
    display(results)
        
    return results

    
    
###################################################    
# Function to be used for testing the SelectKBest
###################################################
def fit_pipe_SelectKBest(X, y, pipe, verbose=False, sample_weight=None ):
    
    gs_results = []  
           
    # For each train/set size run different random splitting
    for run_idx in range(10):

        for k in range(3,20):

            # Split sets        
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.4, random_state=run_idx, stratify=y)

            # Set parameter 
            pipe.set_params(ft__k=k)

            if verbose==True:
                pipe.fit(X_tr, y_tr, verbose=True)
            else:
                pipe.fit(X_tr, y_tr)


            # Evaluate predictions
            y_pred_tr = pipe.predict(X_tr)  
            y_pred_val = pipe.predict(X_val) 
            
            
            # Get name of features from last preprocessing step
            total_features = pipe.named_steps['ip_encoding'].get_features_name().to_list()
            # Get features selected
            features_selected = pipe.named_steps['ft'].get_support()

            res = list(compress(total_features, features_selected))

            # Store results
            gs_results.append({
                'run_idx': run_idx,
                'k':k,                     
                'f1_tr': metrics.f1_score(y_tr, y_pred_tr, average='weighted', sample_weight=sample_weight),
                'f1_val': metrics.f1_score(y_val, y_pred_val, average='weighted', sample_weight=sample_weight),  
                'y_tr': y_tr,
                'y_pred_tr': y_pred_tr,                 
                'y_val': y_val,
                'y_pred_val': y_pred_val,  
                'features': res
            }) 


    gs_results = pd.DataFrame(gs_results)

    return gs_results

##########################################################
# Average results for SelectkBEst
##########################################################
def average_results_SelectKBest(df):
    
    test_size = 0.4
    
    results = []
    
        
    for k in range(3,20): 
        
        df_lock = df.loc[df['k']==k]
        
        f1_tr_mean = df_lock['f1_tr'].mean()
        f1_tr_std = df_lock['f1_tr'].std()
        f1_val_mean = df_lock['f1_val'].mean()
        f1_val_std = df_lock['f1_val'].std()        

        # Average all random states and display results
        results.append({
            'k':k,
            'f1_tr_mean': f1_tr_mean,
            'f1_tr_std': f1_tr_std,
            'f1_val_mean': f1_val_mean,
            'f1_val_std': f1_val_std,           

        })
        
    results = pd.DataFrame(results)      
    
            
    # Find best kappa from validation data
    best_kappa_idx = results['f1_val_mean'].idxmax(axis=1)
    best_kappa = results.loc[best_kappa_idx, 'k']      

    plt.errorbar(results['k'], results['f1_tr_mean'], yerr=results['f1_tr_std'], xerr=None, label='training data')
    plt.errorbar(results['k'], results['f1_val_mean'], yerr=results['f1_val_std'], xerr=None, label='validation data')
    plt.scatter(best_kappa, results['f1_val_mean'].max(), marker='x', c='red', zorder=10)
    plt.title('Training - Validation: {:.0f}% - {:.0f}%, best kappa:{:.0f}'.format((1-test_size)*100, test_size*100, best_kappa), fontsize=14)
    plt.legend(bbox_to_anchor=(1.4, 1.0))     
    plt.xlabel('number of top features (k)', fontsize=12)
    plt.ylabel('avg f1-weighted', fontsize=12)
    plt.show()
    
    df_loc = df.loc[df['k']==best_kappa]
        
    # Confusion matrix for best k
    conf_matrix_tr = []
    conf_matrix_val = []     

    # Train data
    df_ytr = pd.DataFrame(df_loc['y_tr'].tolist())
    df_ytr = df_ytr.T

    df_pred_ytr = pd.DataFrame(df_loc['y_pred_tr'].tolist())
    df_pred_ytr = df_pred_ytr.T

    for c in df_ytr.columns:
        matrix =  metrics.confusion_matrix(y_true=df_ytr[c], y_pred=df_pred_ytr[c])
        conf_matrix_tr.append(matrix)  


    # Validation data
    df_yval = pd.DataFrame(df_loc['y_val'].tolist())
    df_yval = df_yval.T

    df_pred_yval = pd.DataFrame(df_loc['y_pred_val'].tolist())
    df_pred_yval = df_pred_yval.T

    for c in df_yval.columns:
        matrix =  metrics.confusion_matrix(y_true=df_yval[c], y_pred=df_pred_yval[c])
        conf_matrix_val.append(matrix)    


    ##################################
    # Get average confusion matrices
    ##################################
    labels = ['Benign', 'HPortScan', 'C&C', 'DDoS', 'Attack', 'VPortScan']

    # Average confusion matrices and normalize them
    conf_matrix_tr_mean = np.average(conf_matrix_tr, axis=0)
    conf_matrix_tr_mean_normalized = conf_matrix_tr_mean/ conf_matrix_tr_mean.sum(axis=1)[:, np.newaxis]

    conf_matrix_val_mean = np.average(conf_matrix_val, axis=0)
    conf_matrix_val_mean_normalized = conf_matrix_val_mean/ conf_matrix_val_mean.sum(axis=1)[:, np.newaxis]     

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    sns.heatmap(conf_matrix_tr_mean_normalized, annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels, ax=ax[0])
    ax[0].set_ylabel('Actual')
    ax[0].set_xlabel('Predicted')
    ax[0].set_title('Training data, size: {:.0f}%, best kappa:{:.0f}'.format((1-test_size)*100, best_kappa))

    sns.heatmap(conf_matrix_val_mean_normalized, annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels, ax=ax[1])
    ax[1].set_ylabel('Actual')
    ax[1].set_xlabel('Predicted')  
    ax[1].set_title('Validation data, size: {:.0f}%, best kappa:{:.0f}'.format((test_size)*100, best_kappa))

    plt.tight_layout()
    plt.show(block=False)        
    
    res_best_kappa = results.loc[results['k']==best_kappa]
    
    display(res_best_kappa)
    
    return res_best_kappa
        


    
    
