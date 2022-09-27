import os
import re
import pandas as pd
#import modin.pandas as pd

###########################################################
# Funtion to retrieve log paths from different directories
###########################################################
def get_logs():
    '''
    Get all log paths into a list
    '''
    
    # Define filepaths
    data_path = 'data'
    opt_path = 'opt/Malware-Project/BigDataset/IoTScenarios/'
    
    # Define a list to store the paths for the log files
    log_files_benign = []
    log_files_malware = []

    # Define root directory
    root_dir = os.path.join(data_path, opt_path)

    # Get log files
    for dir_ in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, dir_)
        
        if  (sub_dir.startswith(root_dir +  'CTU-Honeypot')) :
            
            for sub_sub_dir in os.listdir(sub_dir):
                full_path = os.path.join(sub_dir,sub_sub_dir)

                for log_file in os.listdir(full_path):
                    log_file_full_path = os.path.join(full_path, log_file)
                    log_files_benign.append(log_file_full_path)
                    
        if  (#sub_dir.endswith('Malware-Capture-1-1') or
             sub_dir.endswith('Malware-Capture-3-1') or
             sub_dir.endswith('Malware-Capture-8-1') or
             sub_dir.endswith('Malware-Capture-20-1') or
             sub_dir.endswith('Malware-Capture-21-1') or
             sub_dir.endswith('Malware-Capture-34-1') or
             sub_dir.endswith('Malware-Capture-42-1') or
             sub_dir.endswith('Malware-Capture-44-1')
             #sub_dir.endswith('Malware-Capture-60-1')
             ) :
            
            for sub_sub_dir in os.listdir(sub_dir):
                full_path = os.path.join(sub_dir,sub_sub_dir)

                for log_file in os.listdir(full_path):
                    log_file_full_path = os.path.join(full_path, log_file)
                    log_files_malware.append(log_file_full_path)                    
            
    return log_files_benign, log_files_malware

###########################################################
# Function to read log files and append them to dataframes
###########################################################
def logs_to_dfs():
    '''
    Parse log files into dataframes 
    '''
    
    # Columns of the dataframe
    cols = ['timestamp', 
            'uid', 
            'origin_address', 
            'origin_port', 
            'response_address', 
            'response_port', 
            'protocol', 
            'service',
            'duration',
            'orig_bytes',
            'resp_bytes',
            'conn_state',
            'local_orig',
            'local_resp',
            'missed_bytes',
            'history',
            'orig_pkts',
            'orig_ip_bytes',
            'resp_pkts',
            'resp_ip_bytes',
            'tunnel_parents',
            'label',
            'detailed_label'
           ]    
        
    # Define a list to hold all dataframes
    df_benign = []
    df_malware = []
    
    # Get log paths
    log_files_benign, log_files_malware = get_logs()
    
    # Get file paths of benign files
    for log_file in log_files_benign:
                
        print('Reading:', log_file)
        
        # Read table into dataframe
        df = pd.read_table(log_file, skiprows=8, skipfooter=1, engine='python', sep='\s+', names = cols)
        # append dataframe to list
        df_benign.append(df)
        
    # Read malware files
    for log_file in log_files_malware:   
        
        print('Reading:', log_file)
        
        # Read table into dataframe
        df = pd.read_table(log_file, skiprows=8, skipfooter=1, engine='python', sep='\s+', names = cols)
        # append dataframe to list
        df_malware.append(df)        
            
    return df_benign, df_malware 

