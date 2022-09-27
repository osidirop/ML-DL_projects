from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import label_binarize


###################################
# Baseline
###################################
def fit_baseline(X, y, strategy):
    
    results = []
    
    dummy_clf = DummyClassifier(strategy=strategy)

    # For each train/set size run different random splitting
    for run_idx in range(10):

        # Split sets        
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.4, random_state=run_idx, stratify=y)

        dummy_clf.fit(X_tr, y_tr)

        y_pred_tr = dummy_clf.predict(X_tr)
        y_pred_val = dummy_clf.predict(X_val)

        results.append({
            'run_idx': run_idx,
            'f1_tr': metrics.f1_score(y_tr, y_pred_tr, average='weighted'),
            'f1_val': metrics.f1_score(y_val, y_pred_val, average='weighted'),
            'acc_tr':metrics.balanced_accuracy_score(y_tr, y_pred_tr),
            'acc_val':metrics.balanced_accuracy_score(y_val, y_pred_val),
            'y_tr': y_tr,
            'y_pred_tr': y_pred_tr,                 
            'y_val': y_val,
            'y_pred_val': y_pred_val, 

        })
            
    results = pd.DataFrame(results)  
    
    return results     


##########################################################
# Average results
##########################################################
def average_results(df):
    
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
        'acc_tr_mean': df['acc_tr'].mean(),
        'acc_tr_std': df['acc_tr'].std(),
        'acc_val_mean': df['acc_val'].mean(),
        'acc_val_std': df['acc_val'].std(),        

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

##########################################################
# Plot validation curves for RandomForestClassifier
##########################################################
def plot_val_curves(cv_results, best_estimator, y_lim_min, y_lim_max, best_param_name=''):
    
    # Convert dictionary to dataframe
    results = pd.DataFrame(cv_results)
    
    full_param_name = 'param_'+ best_param_name
      
    # Get values for x-axis    
    X = results[full_param_name]
        

    # Get best estimator value
    best_n_estim = best_estimator[best_param_name]
    
    # Get for best estimator the mean and std of metrics    
    mean_acc = results.loc[results[full_param_name]==best_n_estim]['mean_test_balanced_accuracy'].values[0]
    mean_f1 = results.loc[results[full_param_name]==best_n_estim]['mean_test_f1_weighted'].values[0]
    mean_precision = results.loc[results[full_param_name]==best_n_estim]['mean_test_precision_weighted'].values[0]
    mean_recall = results.loc[results[full_param_name]==best_n_estim]['mean_test_recall_weighted'].values[0]
    
    std_acc = results.loc[results[full_param_name]==best_n_estim]['std_test_balanced_accuracy'].values[0]
    std_f1 = results.loc[results[full_param_name]==best_n_estim]['std_test_f1_weighted'].values[0]
    std_precision = results.loc[results[full_param_name]==best_n_estim]['std_test_precision_weighted'].values[0]
    std_recall = results.loc[results[full_param_name]==best_n_estim]['std_test_recall_weighted'].values[0]  
    
    gs_results = ({
         best_param_name: best_n_estim,
        'mean_val_acc': mean_acc,
        'std_val_acc': std_acc,
        'mean_val_f1': mean_f1,
        'std_val_f1': std_f1,
        'mean_val_precision': mean_precision,
        'std_val_precision': std_precision,
        'mean_val_recall': mean_recall,
        'std_val_recall': std_recall
    })

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))    
    ax[0,0].plot(X, results['mean_train_balanced_accuracy'], label='train', color='tab:blue')
    ax[0,0].plot(X, results['mean_test_balanced_accuracy'], label='validation', color='tab:orange')
    ax[0,0].scatter(best_n_estim, mean_acc, marker='x', c='red', zorder=10, label='best '+ best_param_name + ':{:.2f}'.format(best_n_estim) )
    ax[0,0].set_xlabel(best_param_name, fontsize=13)
    ax[0,0].set_ylabel('mean balanced accuracy', fontsize=13)
    ax[0,0].set_title('Validation curves with 3 StratifiedKFold splits', fontsize=14)
    ax[0,0].set_ylim(y_lim_min, y_lim_max)
    ax[0,0].legend(loc='best')
    
    ax[0,1].plot(X, results['mean_train_f1_weighted'], label='train', color='tab:blue')
    ax[0,1].plot(X, results['mean_test_f1_weighted'], label='validation', color='tab:orange')
    ax[0,1].scatter(best_n_estim, mean_f1, marker='x', c='red', zorder=10, label='best '+ best_param_name + ':{:.2f}'.format(best_n_estim))
    ax[0,1].set_xlabel(best_param_name, fontsize=13)
    ax[0,1].set_ylabel('mean f1-weighted score', fontsize=13)
    ax[0,1].set_title('Validation curves with 3 StratifiedKFold splits', fontsize=14) 
    ax[0,1].set_ylim(y_lim_min, y_lim_max)
    ax[0,1].legend(loc='best')
    
    ax[1,0].plot(X, results['mean_train_precision_weighted'], label='train', color='tab:blue')
    ax[1,0].plot(X, results['mean_test_precision_weighted'], label='validation', color='tab:orange')
    ax[1,0].scatter(best_n_estim, mean_precision, marker='x', c='red', zorder=10, label='best '+ best_param_name + ':{:.2f}'.format(best_n_estim))
    ax[1,0].set_xlabel(best_param_name, fontsize=13)
    ax[1,0].set_ylabel('mean precision weighted', fontsize=13)
    ax[1,0].set_title('Validation curves with 3 StratifiedKFold splits', fontsize=14) 
    ax[1,0].set_ylim(y_lim_min, y_lim_max)
    ax[1,0].legend(loc='best')
    
    ax[1,1].plot(X, results['mean_train_recall_weighted'], label='train', color='tab:blue')
    ax[1,1].plot(X, results['mean_test_recall_weighted'], label='validation', color='tab:orange')
    ax[1,1].scatter(best_n_estim, mean_recall, marker='x', c='red', zorder=10, label='best '+ best_param_name + ':{:.2f}'.format(best_n_estim))
    ax[1,1].set_xlabel(best_param_name, fontsize=13)
    ax[1,1].set_ylabel('mean recall weighted', fontsize=13)
    ax[1,1].set_title('Validation curves with 3 StratifiedKFold splits', fontsize=14) 
    ax[1,1].set_ylim(y_lim_min, y_lim_max)
    ax[1,1].legend(loc='best')
    
    
    plt.tight_layout()
    plt.show()    
    
    gs_results = pd.DataFrame(gs_results, index=[0])
    display(gs_results)
    
    return gs_results

##########################################################
# Results for test data
##########################################################
def get_results_for_test_data(y_te, y_pred, y_te_score, classifier=''):
    
    # Define labels
    labels = ['Benign', 'HPortScan', 'C&C', 'DDoS', 'Attack', 'VPortScan']    
    
    ##########################################
    # Confusion Matrix
    ##########################################
    
    # Get confusion matrix and normalize it
    matrix =  metrics.confusion_matrix(y_true=y_te, y_pred=y_pred)
    matrix_mean_norm = matrix/ matrix.sum(axis=1)[:, np.newaxis]    

    # Plot normalized confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(15,4))

    sns.heatmap(matrix_mean_norm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, ax=ax[0])
    ax[0].set_ylabel('Actual', fontsize=13)
    ax[0].set_xlabel('Predicted', fontsize=13)
    ax[0].set_title('Confusion matrix for test data\n' + classifier, fontsize=14)
 
    ##########################################
    # ROC-AUC curves
    ##########################################
    
#     if classifier =='LogisticRegression':
    
    # Calculate ROC-AUC curves and plot
    y_te_bin = label_binarize(y_te, classes=[0, 1, 2, 3, 4, 5])
    n_classes = y_te_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    cm = plt.get_cmap('tab10')
    colorrange = np.arange(0,10)/10

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_te_bin[:, i], y_te_score[:, i])
        auc = metrics.auc(fpr[i], tpr[i])
        ax[1].plot(fpr[i], tpr[i], color=cm(colorrange[i]), lw=2, label=labels[i]+' (AUC:{:.2f})'.format(auc))

    ax[1].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate', fontsize=13)
    ax[1].set_ylabel('True Positive Rate', fontsize=13)
    ax[1].set_title('ROC curves for test data \n'+classifier, fontsize=14)
    ax[1].legend(bbox_to_anchor=(1, 1), fontsize=13)

    plt.tight_layout()
    plt.show()   
        
#     else:
        
#         ax[1].set_visible(False)
    
    ##########################################
    # Classification report
    ##########################################
    print('\n\n')
    print('\033[1m Classification report for test data (\033[0m' + '\033[1m'+ classifier + ')\033[0m \n')
    cl_report = metrics.classification_report(y_te, y_pred, target_names=labels, output_dict=False)
    print(cl_report)
    
    # Get classification report as dataframe to return
    cl_report = metrics.classification_report(y_te, y_pred, target_names=labels, output_dict=True)
    cl_report = pd.DataFrame(cl_report)    
    
    return cl_report




#######################################################
# Compare results notebook 4
#######################################################
def compare_classifiers_all(baseline_results_avg, scores, fig_h, fig_w):
    
    fig, ax = plt.subplots(2, 2, figsize=(fig_h, fig_w))

    # Define colors
    cm = plt.get_cmap('tab10')
    colorrange = np.arange(0,10)/10

    ##############################################
    # Plot all together
    ##############################################

    ########################## Training set ############################
    # From results from baseline output
    ax[0,0].bar('baseline', baseline_results_avg['acc_tr_mean'].values[0], 
                        yerr=baseline_results_avg['acc_tr_std'].values[0], color=cm(colorrange[0]), 
                        label='baseline')

    # Plot results from models  
    i = 1
    for idx in scores.index:

        ax[0,0].bar(idx, scores.loc[idx,'tr_balanced_accuracy_mean'], 
                     yerr=scores.loc[idx,'tr_balanced_accuracy_std'], color=cm(colorrange[i]), 
                     label=idx)
        i = i +1

    plt.setp(ax[0,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax[0,0].tick_params(axis='x', labelrotation = 45)
    ax[0,0].set_title('Comparison of different models with the baseline \n Training set ', fontsize=15)
    ax[0,0].set_ylabel('mean balanced train accuracy', fontsize=13)
    ax[0,0].grid()
    
    ########################## Validation set ############################
    # From results from baseline output
    ax[0,1].bar('baseline', baseline_results_avg['acc_val_mean'].values[0], 
                        yerr=baseline_results_avg['acc_val_std'].values[0], color=cm(colorrange[0]), 
                        label='baseline')

    # Plot results from models  
    i = 1
    for idx in scores.index:

        ax[0,1].bar(idx, scores.loc[idx,'val_balanced_accuracy_mean'], 
                     yerr=scores.loc[idx,'val_balanced_accuracy_std'], color=cm(colorrange[i]), 
                     label=idx)
        i = i +1

    plt.setp(ax[0,1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax[0,1].tick_params(axis='x', labelrotation = 45)
    ax[0,1].set_title('Comparison of different models with the baseline \n Validation set ', fontsize=15)
    ax[0,1].set_ylabel('mean balanced valid accuracy', fontsize=13)
    ax[0,1].grid()    

    ##############################################
    # Plot only models
    ##############################################
    
   ########################## Training set ############################    

    i = 1
    for idx in scores.index:

        ax[1,0].bar(idx, scores.loc[idx,'tr_balanced_accuracy_mean'], 
                     yerr=scores.loc[idx,'tr_balanced_accuracy_std'], color=cm(colorrange[i]), 
                     label=idx)
        i = i +1

    plt.setp(ax[1,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax[1,0].set_title('Comparison of different models \n Training set', fontsize=15)
    ax[1,0].set_ylabel('mean balanced train accuracy', fontsize=13)
    ax[1,0].set_ylim(0.94,1.02)
    ax[1,0].grid()

   ########################## Validation set ############################    

    i = 1
    for idx in scores.index:

        ax[1,1].bar(idx, scores.loc[idx,'val_balanced_accuracy_mean'], 
                     yerr=scores.loc[idx,'val_balanced_accuracy_std'], color=cm(colorrange[i]), 
                     label=idx)
        i = i +1

    plt.setp(ax[1,1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax[1,1].set_title('Comparison of different models \n Validation set', fontsize=15)
    ax[1,1].set_ylabel('mean balanced valid accuracy', fontsize=13)
    ax[1,1].set_ylim(0.94,1.02)
    ax[1,1].grid()

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
            'precision_tr': metrics.precision_score(y_tr, y_pred_tr, average='weighted', sample_weight=sample_weight),
            'precision_val': metrics.precision_score(y_val, y_pred_val, average='weighted', sample_weight=sample_weight),  
            'recall_tr': metrics.recall_score(y_tr, y_pred_tr, average='weighted', sample_weight=sample_weight),
            'recall_val': metrics.recall_score(y_val, y_pred_val, average='weighted', sample_weight=sample_weight),    
            'acc_tr': metrics.balanced_accuracy_score(y_tr, y_pred_tr),
            'acc_val': metrics.balanced_accuracy_score(y_val, y_pred_val),             
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
        # f1-score
        'f1_tr_mean': df['f1_tr'].mean(),
        'f1_tr_std': df['f1_tr'].std(),
        'f1_val_mean': df['f1_val'].mean(),
        'f1_val_std': df['f1_val'].std(),    
         # precision
        'precision_tr_mean': df['precision_tr'].mean(),
        'precision_tr_std': df['precision_tr'].std(),   
        'precision_val_mean': df['precision_val'].mean(),
        'precision_val_std': df['precision_val'].std(),   
         # recall
        'recall_tr_mean': df['recall_tr'].mean(),
        'recall_tr_std': df['recall_tr'].std(),   
        'recall_val_mean': df['recall_val'].mean(),
        'recall_val_std': df['recall_val'].std(),        
         # balanced accuracy
        'acc_tr_mean': df['acc_tr'].mean(),
        'acc_tr_std': df['acc_tr'].std(),   
        'acc_val_mean': df['acc_val'].mean(),
        'acc_val_std': df['acc_val'].std(),         

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
    display(results.T)
        
    return results