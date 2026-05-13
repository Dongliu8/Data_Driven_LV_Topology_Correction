



import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
V = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_1.csv")


nn_mising = 0.0

acc_ave = []

for ii in range(3): # we run the simulation 3 and calculation
    
    # 3 switches---------------------------------------------------------------
    # training dataset loading
    data1 = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_1.csv") # need you put the excel or csv datasets in the same folders
    data1 = data1.iloc[1:, 1:]
    data1 = data1.to_numpy(dtype=np.float64)

    data2 = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_2.csv") # need you put the excel or csv datasets in the same folders
    data2 = data2.iloc[1:, 1:]
    data2 = data2.to_numpy(dtype=np.float64)
    
    data3 = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_3.csv") # need you put the excel or csv datasets in the same folders
    data3 = data3.iloc[1:, 1:]
    
    #  larger training datasets
    # data2 = data2.iloc[2:, 1:]
    data3 = data3.to_numpy(dtype=np.float64)
    
    
    # testing dataset loading---------------------------------
    data11 = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_11.csv", header=None) # need you put the excel or csv datasets in the same folders
    data11 = data11.iloc[2:, 1:]
    data11 = data11.to_numpy(dtype=np.float64)
       
    data21 = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_21.csv", header=None) # need you put the excel or csv datasets in the same folders
    data21 = data21.iloc[2:, 1:]
    data21 = data21.to_numpy(dtype=np.float64)
    
    data31 = pd.read_csv(BASE_DIR / "synthetic_simulated_filtered_31.csv", header=None) # need you put the excel or csv datasets in the same folders
    data31 = data31.iloc[2:, 1:]
    data31 = data31.to_numpy(dtype=np.float64)
    
    
    #  index of the available SM on each cable----------------
    # extracting indexs: 2
    extract_index = [47,48,
                      96,97,
                      145,146]
    
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
    
           

    
    # 4. Normalize (if necessary)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.ravel()
    
    # 5. model training-------------------------------------RF
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
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
    # clf = LGBMClassifier(n_estimators=100, random_state=42)
    # clf.fit(X_train, y_train)
    
    
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







































