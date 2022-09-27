import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

    
       
##############################################
# Function to get all results together
##############################################

def get_all_results():

    ls_models  = ['GaussianNB', 'KNeighborsClassifier', 'RandomForestClassifier', 'LinearSVC', 
                  'SVC', 'LogisticRegression_OvR', 'LogisticRegression_softmax', 'RidgeClassifier']

    ls_empty = [0]*len(ls_models)              

    df_all = ({'Classifier': ls_models, 
              'f1': ls_empty,
              'precision': ls_empty,
              'recall': ls_empty,
              'accuracy': ls_empty,              
              })

    df_all = pd.DataFrame(df_all).set_index('Classifier')


    # 1. SVC 
    df_svc_tuned = pd.read_csv(os.path.join('results', 'gs_results_best_crossval_svc.csv'))
    df_all.loc['SVC', 'f1'] = df_svc_tuned['mean_val_f1'].iloc[0]
    df_all.loc['SVC', 'precision'] = df_svc_tuned['mean_val_precision'].iloc[0]
    df_all.loc['SVC', 'recall'] = df_svc_tuned['mean_val_recall'].iloc[0]
    df_all.loc['SVC', 'accuracy'] = df_svc_tuned['mean_val_acc'].iloc[0]

    # 2. LogisticRegression_ovr 
    df_logreg_tuned = pd.read_csv(os.path.join('results', 'gs_results_best_crossval_logreg.csv'))
    df_all.loc['LogisticRegression_OvR', 'f1'] = df_logreg_tuned['mean_val_f1'].iloc[0]
    df_all.loc['LogisticRegression_OvR', 'precision'] = df_logreg_tuned['mean_val_precision'].iloc[0]
    df_all.loc['LogisticRegression_OvR', 'recall'] = df_logreg_tuned['mean_val_recall'].iloc[0]
    df_all.loc['LogisticRegression_OvR', 'accuracy'] = df_logreg_tuned['mean_val_acc'].iloc[0]


     # 3. GaussianNB 
    df_nb = pd.read_csv(os.path.join('results', 'results_avg_GaussianNB.csv'))
    df_all.loc['GaussianNB', 'f1'] = df_nb['f1_val_mean'].iloc[0]
    df_all.loc['GaussianNB', 'precision'] = df_nb['precision_val_mean'].iloc[0]
    df_all.loc['GaussianNB', 'recall'] = df_nb['recall_val_mean'].iloc[0]
    df_all.loc['GaussianNB', 'accuracy'] = df_nb['acc_val_mean'].iloc[0]

    # 4. KNeighborsClassifier 
    df_knn = pd.read_csv(os.path.join('results', 'results_avg_KNeighborsClassifier.csv'))
    df_all.loc['KNeighborsClassifier', 'f1'] = df_knn['f1_val_mean'].iloc[0]
    df_all.loc['KNeighborsClassifier', 'precision'] = df_knn['precision_val_mean'].iloc[0]
    df_all.loc['KNeighborsClassifier', 'recall'] = df_knn['recall_val_mean'].iloc[0]
    df_all.loc['KNeighborsClassifier', 'accuracy'] = df_knn['acc_val_mean'].iloc[0]

    # 5. RandomForestClassifier
    df_rfc = pd.read_csv(os.path.join('results', 'results_avg_RandomForestClassifier.csv'))
    df_all.loc['RandomForestClassifier', 'f1'] = df_rfc['f1_val_mean'].iloc[0]
    df_all.loc['RandomForestClassifier', 'precision'] = df_rfc['precision_val_mean'].iloc[0]
    df_all.loc['RandomForestClassifier', 'recall'] = df_rfc['recall_val_mean'].iloc[0]
    df_all.loc['RandomForestClassifier', 'accuracy'] = df_rfc['acc_val_mean'].iloc[0]

    # 6. LinearSVC
    df_svc_lin = pd.read_csv(os.path.join('results', 'results_avg_LinearSVC.csv'))
    df_all.loc['LinearSVC', 'f1'] = df_svc_lin['f1_val_mean'].iloc[0]
    df_all.loc['LinearSVC', 'precision'] = df_svc_lin['precision_val_mean'].iloc[0]
    df_all.loc['LinearSVC', 'recall'] = df_svc_lin['recall_val_mean'].iloc[0]
    df_all.loc['LinearSVC', 'accuracy'] = df_svc_lin['acc_val_mean'].iloc[0]

    # 7. LogisticRegression_softmax
    df_logreg_soft = pd.read_csv(os.path.join('results', 'results_avg_LogisticRegression_softmax.csv'))
    df_all.loc['LogisticRegression_softmax', 'f1'] = df_logreg_soft['f1_val_mean'].iloc[0]
    df_all.loc['LogisticRegression_softmax', 'precision'] = df_logreg_soft['precision_val_mean'].iloc[0]
    df_all.loc['LogisticRegression_softmax', 'recall'] = df_logreg_soft['recall_val_mean'].iloc[0]
    df_all.loc['LogisticRegression_softmax', 'accuracy'] = df_logreg_soft['acc_val_mean'].iloc[0]

    # 8. RidgeClassifier
    df_ridge = pd.read_csv(os.path.join('results', 'results_avg_RidgeClassifier.csv'))
    df_all.loc['RidgeClassifier', 'f1'] = df_ridge['f1_val_mean'].iloc[0]
    df_all.loc['RidgeClassifier', 'precision'] = df_ridge['precision_val_mean'].iloc[0]
    df_all.loc['RidgeClassifier', 'recall'] = df_ridge['recall_val_mean'].iloc[0]
    df_all.loc['RidgeClassifier', 'accuracy'] = df_ridge['acc_val_mean'].iloc[0]

    display(df_all)     
    print('\n')
    
    # Plot
    df_all.plot(lw=0, marker=".", markersize=12, figsize=(8, 5))
    plt.xticks(rotation = 45, ha='right', rotation_mode='anchor')
    plt.title('Summary plot from the validation data',fontsize=15)
    plt.xlabel('classifiers', fontsize=13)
    plt.ylabel('mean scores/accuracy', fontsize=13)
    plt.legend(bbox_to_anchor=(1.3, 1), fontsize=13)
    plt.grid()
    plt.show()    

   


    #return df_all

############################################


##############################################
# Function to save test accuracy in a file
##############################################
def save_test_accuracy(filename, model, acc):
    
    '''
    Function to save test accuracy
    '''
    
    d = {'model': model, 'test_accuracy': acc}
    
    with open(filename, 'w') as f: 
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)