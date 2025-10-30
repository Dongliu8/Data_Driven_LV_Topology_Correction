
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:15:31 2025

@author: dliu8
"""


# =============================================================================
# successfully on 10/03/2025---reference
# =============================================================================


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



nn_mising = 0.0

acc_ave = []

for ii in range(3): # we run the simulation 3 and calculation
    
    # 3 switches---------------------------------------------------------------
    # training dataset loading
    # data1 = pd.read_csv(r'switching_state_set\three_phases\multiple_per_node/simulated_filtered_0.csv')
    data1 = pd.read_csv(r'switching_state_set\three_phases\multiple_per_node/simulated_filtered_12.csv')
    data1 = data1.iloc[1:, 1:]
    #  larger training datasets
    # data1 = pd.read_csv(r'switching_state_set\three_phases\multiple_per_node/simulated_filtered_00.csv',header=None)
    # data1 = data1.iloc[2:, 1:]
    data1 = data1.to_numpy(dtype=np.float64)

    # data2 = pd.read_csv(r'switching_state_set_different_opening\three_phases\multiple_per_node/simulated_filtered_0.csv')
    data2 = pd.read_csv(r'switching_state_set_different_opening\three_phases\multiple_per_node/simulated_filtered_12.csv')
    data2 = data2.iloc[1:, 1:]
    #  larger training datasets
    # data2 = pd.read_csv(r'switching_state_set_different_opening\three_phases\multiple_per_node/simulated_filtered_00.csv',header=None)
    # data2 = data2.iloc[2:, 1:]
    data2 = data2.to_numpy(dtype=np.float64)
    
    # data3 = pd.read_csv(r'switching_state_set_no_opening\three_phases\multiple_per_node/simulated_filtered_0.csv')
    data3 = pd.read_csv(r'switching_state_set_no_opening\three_phases\multiple_per_node/simulated_filtered_12.csv')
    data3 = data3.iloc[1:, 1:]
    #  larger training datasets
    # data2 = data2.iloc[2:, 1:]
    data3 = data3.to_numpy(dtype=np.float64)
    
    
    # testing dataset loading---------------------------------
    data11 = pd.read_csv(r'switching_state_set\three_phases\multiple_per_node/simulated_filtered_14.csv',header=None)
    data11 = data11.iloc[2:, 1:]
    data11 = data11.to_numpy(dtype=np.float64)
       
    data21 = pd.read_csv(r'switching_state_set_different_opening\three_phases\multiple_per_node/simulated_filtered_14.csv',header=None)
    data21 = data21.iloc[2:, 1:]
    data21 = data21.to_numpy(dtype=np.float64)
    
    data31 = pd.read_csv(r'switching_state_set_no_opening\three_phases\multiple_per_node/simulated_filtered_14.csv',header=None)
    data31 = data31.iloc[2:, 1:]
    data31 = data31.to_numpy(dtype=np.float64)
    
    
    #  index of the available SM on each cable----------------
    # extracting indexs: 2
    # extract_index = [47,48,
    #                   96,97,
    #                   145,146]
    
    # extracting indexs: 10
    extract_index = [39,40,41,42,43,44,45,46,47,48,
                      88,89,90,91,92,93,94,95,96,97,
                      137,138,139,140,141,142,143,144,145,146]
    
    # extract_index0 = [30,35,25,42,28,15,45,46,47,48,
    #                   55,89,50,91,49,60,94,85,76,97,
    #                   119,121,139,130,110,140,133,144,111,146]
    
    # # # extracting indexs: 20
    # extract_index = [29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    #                   78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,
    #                   127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146]
    
    
    # extract SM from the available SM -----------------------------------------
    data1 = np.array(data1)[extract_index]
    data1 = np.delete(data1, np.random.choice(data1.shape[1], int(data1.shape[1] * nn_mising), replace=False), axis=1)
        
    data2 = np.array(data2)[extract_index]
    data2 = np.delete(data2, np.random.choice(data2.shape[1], int(data2.shape[1] * nn_mising), replace=False), axis=1)
    
    data3 = np.array(data3)[extract_index]
    data3 = np.delete(data3, np.random.choice(data3.shape[1], int(data3.shape[1] * nn_mising), replace=False), axis=1)
    
    # data0 = np.concatenate((data1, data2), axis=1)  # axis=1 
    data0 = np.concatenate((data1, data2, data3), axis=1)  # axis=1 
    data0 = data0.T
    # label = [0] * data1.shape[1] + [1] * data2.shape[1]

    label = [0] * data1.shape[1] + [1] * data2.shape[1] + [2] * data3.shape[1] 
    label = np.array(label).reshape(-1, 1)
    data = np.concatenate((data0, label), axis=1)
    
    # ------------------------------------------
    # X_train, X_test, y_train, y_test = train_test_split(data0, label, test_size=0.2, random_state=42)
    # y_train = y_train.astype(int)  # convert it "int"
    # y_test  = y_test.astype(int)  # convert it "int"
    
    
    # -----------------------------------
    data11 = np.array(data11)[extract_index]
    data11 = np.delete(data11, np.random.choice(data11.shape[1], int(data11.shape[1] * nn_mising), replace=False), axis=1)
        
    data21 = np.array(data21)[extract_index]
    data21 = np.delete(data21, np.random.choice(data21.shape[1], int(data21.shape[1] * nn_mising), replace=False), axis=1)
    
    data31 = np.array(data31)[extract_index]
    data31 = np.delete(data31, np.random.choice(data31.shape[1], int(data31.shape[1] * nn_mising), replace=False), axis=1)
    
    # data_test = np.concatenate((data11, data21), axis=1)  # axis=1 
    data_test = np.concatenate((data11, data21, data31), axis=1)  # axis=1 
    data_test = data_test.T
    # label_test = [0] * data1.shape[1] + [1] * data2.shape[1]  

    label_test = [0] * data11.shape[1] + [1] * data21.shape[1] + [2] * data31.shape[1] 
    label_test = np.array(label_test).reshape(-1, 1)
    data_test = np.concatenate((data_test, label_test), axis=1)
    
    
    X_train = data[:, :-1]         # features of training datasets
    y_train = data[:, -1]          # labels of training datasets
    X_test = data_test[:, :-1]     # features of testing datasets 
    y_test = data_test[:, -1]      # labels of the testing datasets
    y_train = y_train.astype(int)  #  int
    y_test  = y_test.astype(int)   #  int
    
       
    
    # # # 5 switches,------------------------------------------------------------------------
    # # =============================================================================
    # data1 = pd.read_csv(r'switching_state_set_five_cables\three_phases\multiple_per_node/simulated_0.csv')
    # data1 = np.array(data1)
    # data1 = data1[1:,1:]
       
    # data2 = pd.read_csv(r'switching_state_set_five_cables_2\three_phases\multiple_per_node/simulated_0.csv')
    # data2 = np.array(data2)
    # data2 = data2[1:,1:]
    
    # data3 = pd.read_csv(r'switching_state_set_five_cables_3\three_phases\multiple_per_node/simulated_0.csv')
    # data3 = np.array(data3)
    # data3 = data3[1:,1:]
    
    # data4 = pd.read_csv(r'switching_state_set_five_cables_4\three_phases\multiple_per_node/simulated_0.csv')
    # data4 = np.array(data4)
    # data4 = data4[1:,1:]
    
    
    # # testing datasets
    # data11 = pd.read_csv(r'switching_state_set_five_cables\three_phases\multiple_per_node/simulated_filtered_4.csv')
    # data11 = np.array(data11)
    # data11 = data11[1:,1:]
        
    # data21 = pd.read_csv(r'switching_state_set_five_cables_2\three_phases\multiple_per_node/simulated_filtered_4.csv')
    # data21 = np.array(data21)
    # data21 = data21[1:,1:]
    
    # data31 = pd.read_csv(r'switching_state_set_five_cables_3\three_phases\multiple_per_node/simulated_filtered_4.csv')
    # data31 = np.array(data31)
    # data31 = data31[1:,1:]
    
    # data41 = pd.read_csv(r'switching_state_set_five_cables_4\three_phases\multiple_per_node/simulated_filtered_4.csv')
    # data41 = np.array(data41)
    # data41 = data41[1:,1:]
     
    
    # # extracting indexs: 2
    # extract_index = [47,48,
    #                   96,97,
    #                   145,146,
    #                   194,195,
    #                   243,244]
    
    # extracting indexs: 10
    # extract_index = [39,40,41,42,43,44,45,46,47,48,
    #                   88,89,90,91,92,93,94,95,96,97,
    #                   137,138,139,140,141,142,143,144,145,146,
    #                   186,187,188,189,190,191,192,193,194,195,
    #                   235,236,237,238,239,240,241,242,243,244]
    
    # extract_index0 = [30,35,25,42,28,15,45,46,47,48,
    #                   55,89,50,91,49,60,94,85,76,97,
    #                   119,121,139,130,110,140,133,144,111,146,
    #                   150,177,190,165,170,181,155,163,184,195,
    #                   215,216,207,228,229,200,241,232,243,244]
    
    # # extracting indexs: 20
    # extract_index = [29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    #                   78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,
    #                   127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,
    #                   176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,
    #                   225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244]
    
    
    # data1 = np.array(data1)[extract_index]
    # data1 = np.delete(data1, np.random.choice(data1.shape[1], int(data1.shape[1] * nn_mising), replace=False), axis=1)
        
    # # data2 = np.array(data2)[extract_index]
    # data2 = np.delete(data2, np.random.choice(data2.shape[1], int(data2.shape[1] * nn_mising), replace=False), axis=1)
    
    # # data3 = np.array(data3)[extract_index]
    # data3 = np.delete(data3, np.random.choice(data3.shape[1], int(data3.shape[1] * nn_mising), replace=False), axis=1)
    
    # # data4 = np.array(data4)[extract_index]
    # data4 = np.delete(data4, np.random.choice(data4.shape[1], int(data4.shape[1] * nn_mising), replace=False), axis=1)
    
    # data = np.concatenate((data1, data2, data3, data4), axis=1)  # axis=1 表示按列连接
    # data = data.T
    
    # label = [0] * data1.shape[1] + [1] * data2.shape[1] + [3] * data3.shape[1] + [4] * data4.shape[1]
    # label = np.array(label).reshape(-1, 1)
    # data = np.concatenate((data, label), axis=1)
    
    
    # 2. Process the data (assuming the last column is label)
    # X = data[:, :-1]  # features
    # y = data[:, -1]   # labels
    
    # 3. split it into training and testing datasets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # y_train = y_train.astype(int)  #  int
    # y_test  = y_test.astype(int)   #  int
    # -------------------------------------------------
    
    
    # # data11 = np.array(data11)[extract_index]
    # data11 = np.delete(data11, np.random.choice(data11.shape[1], int(data11.shape[1] * nn_mising), replace=False), axis=1)
        
    # # data21 = np.array(data21)[extract_index]
    # data21 = np.delete(data21, np.random.choice(data21.shape[1], int(data21.shape[1] * nn_mising), replace=False), axis=1)
    
    # # data31 = np.array(data31)[extract_index]
    # data31 = np.delete(data31, np.random.choice(data31.shape[1], int(data31.shape[1] * nn_mising), replace=False), axis=1)
    
    # # data41 = np.array(data41)[extract_index]
    # data41 = np.delete(data41, np.random.choice(data41.shape[1], int(data41.shape[1] * nn_mising), replace=False), axis=1)
     
    # -------------------------------------------------
    
    
    # use the data from another SM as the testing apporaches, generalization approaches
    # data1 = pd.read_csv(r'switching_state_set_five_cables\three_phases\multiple_per_node/simulated_0.csv')
    # data1 = np.array(data1)
    # data1 = data1[1:,1:]
       
    # data2 = pd.read_csv(r'switching_state_set_five_cables_2\three_phases\multiple_per_node/simulated_0.csv')
    # data2 = np.array(data2)
    # data2 = data2[1:,1:]
    
    # data3 = pd.read_csv(r'switching_state_set_five_cables_3\three_phases\multiple_per_node/simulated_0.csv')
    # data3 = np.array(data3)
    # data3 = data3[1:,1:]
    
    # data4 = pd.read_csv(r'switching_state_set_five_cables_4\three_phases\multiple_per_node/simulated_0.csv')
    # data4 = np.array(data4)
    # data4 = data4[1:,1:]
    
    # data1 = np.array(data1)[extract_index0]
    # data1 = np.delete(data1, np.random.choice(data1.shape[1], int(data1.shape[1] * nn_mising), replace=False), axis=1)
        
    # data2 = np.array(data2)[extract_index0]
    # data2 = np.delete(data2, np.random.choice(data2.shape[1], int(data2.shape[1] * nn_mising), replace=False), axis=1)
    
    # data3 = np.array(data3)[extract_index0]
    # data3 = np.delete(data3, np.random.choice(data3.shape[1], int(data3.shape[1] * nn_mising), replace=False), axis=1)
    
    # data4 = np.array(data4)[extract_index0]
    # data4 = np.delete(data4, np.random.choice(data4.shape[1], int(data4.shape[1] * nn_mising), replace=False), axis=1)
    
    # data_test = np.concatenate((data1, data2, data3, data4), axis=1)  # axis=1 
    # data_test = data_test.T
    
    # label_test = [0] * data1.shape[1] + [1] * data2.shape[1] + [3] * data3.shape[1] + [4] * data4.shape[1]
    # label_test = np.array(label_test).reshape(-1, 1)
    # data_test = np.concatenate((data_test, label_test), axis=1)
    # -------------------------------------------------
   
    # data_test = np.concatenate((data11, data21, data31, data41), axis=1)  # axis=1 
    # data_test = data_test.T
    
    # label_test = [0] * data11.shape[1] + [1] * data21.shape[1] + [3] * data31.shape[1] + [4] * data41.shape[1]
    # label_test = np.array(label_test).reshape(-1, 1)
    # data_test = np.concatenate((data_test, label_test), axis=1)
    
   
    # X_train = data[:, :-1]  
    # y_train = data[:, -1]  
    # X_test = data_test[:, :-1]
    # y_test = data_test[:, -1]   
    # y_train = y_train.astype(int)  
    # y_test  = y_test.astype(int)  
    
    
    # 4. Normalize (if necessary)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.ravel()
    
    # 5. model training-------------------------------------RF
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X_train, y_train)
    
    # model training-------------------------------------SVM
    # clf = SVC(kernel='rbf', C=1.0, random_state=42)
    # clf.fit(X_train, y_train)
    
    # # model training-------------------------------------GNB
    # clf = GaussianNB()
    # clf.fit(X_train, y_train)
    
    # model training-------------------------------------XGBoost
    # clf = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
    # clf.fit(X_train, y_train)

    # model training-------------------------------------LightGBM
    clf = LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    
    # 6. predicting
    y_pred = clf.predict(X_test)
    
    # 7. evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    acc_ave.append(accuracy)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    

acc_ave = np.array(acc_ave)
print(np.mean(acc_ave))













































