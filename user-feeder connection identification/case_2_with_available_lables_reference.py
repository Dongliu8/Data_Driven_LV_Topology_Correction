





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats                 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import math
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent




nn = 0.50           # percentage of the unknown information
nn_mising = 0.50    # percentage of the random missing data point
k = 5               # parameters of the traditional KNN algorithms
error_level = 0.05  # add error (0.0 %, 0.2 %, 0.5 %, 1 %, 2 %, and 5 %).

for tt in range(1): # to run it multiple times and obtain the avrage 
    # V = your own datasets
    V = pd.read_csv(BASE_DIR /"simulated_voltage_data_0_noise_1.csv") # need you put the excel or csv datasets in the same folders

    
    # 1. 
    cols_to_drop = ['Unnamed: 0', 'timestamp']
    V = V.drop(columns=[c for c in cols_to_drop if c in V.columns])
    
    # 2. 
    V = V.apply(pd.to_numeric, errors='coerce')
    
    # 3. 
    V = V.dropna(axis=0)
    V = V.iloc[1:, :]
    
    # 4. 
    V_matrix = V.values.astype(np.float64) / 230

        
    # 5. 
    print(f"data: {V_matrix.shape}")
    print(f"data: {V_matrix.dtype}")
    
   #-------------------

    noise_matrix = np.random.normal(1, error_level/3, size=V.shape)
    V = V * noise_matrix
  

    #  missing percentage
    num_cols = V.shape[1]
    num_cols_to_delete = int(num_cols * nn_mising)
    cols_to_delete = np.random.choice(num_cols, num_cols_to_delete, replace=False)
    V = np.delete(V, cols_to_delete, axis=1)
    
    # =============================================================================
    
    if len(V) ==63:  # your own labels
        labels0 = np.zeros((V.shape[0], 1), dtype=int)
        labels0[0:43] = 1
        labels0[43:63] = 2

    
    def calculate_accuracy(predicted_labels, true_labels):
      correct_predictions = np.sum(predicted_labels == true_labels)
      total_predictions = len(predicted_labels)
      accuracy = correct_predictions / total_predictions
      return accuracy
    
    def custom_distance(x, y):
        pearson_corr, _ = stats.pearsonr(x, y) 
        pearson_corr = 1 - min(1/4 * math.log(1 + np.exp(1 + 4*pearson_corr)), 1)  # for single-phase data works well in simplifed networks
        return pearson_corr
    
    # correlation calculation 
    pearson_corr = np.zeros((V.shape[0], V.shape[0]))
    for i in range(V.shape[0]):
        for j in range(V.shape[0]):
            pearson_corr[i, j], _ = stats.pearsonr(V[i], V[j]) 
            pearson_corr[i, j] = math.log((1 + pearson_corr[i, j])/((1-pearson_corr[i, j] + 0.5)))  # for single-phase data works well in simplifed networks
        
    
    # =============================================================================
    # suceessfully on 04/11/2024
    # based on pearson_corr
    # =============================================================================
    
    accuracy0 = []
    known_labels = np.array([1] * 43 + [2] * (63 - 43))

    # =============================================================================
    # traditional KNN  suceessfully on 04/11/2024
    # =============================================================================
      
    accuracy0 = []

    
    for i in range(10):    
        labels = known_labels.copy()
        unknown_indices = np.random.choice(np.arange(63), size=int(63 * nn), replace=False)
        labels[unknown_indices] = -1  # Mark some samples as unknown
        
        
        known_indices = labels != -1  
        
    
        X_known = V[known_indices]
        y_known = labels[known_indices]
        X_unknown = V[unknown_indices]
        
        # create KNN model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_known, y_known)
        
        # prediction  of the unknow labels
        predicted_labels = knn.predict(X_unknown)
        labels[unknown_indices] = predicted_labels
        labels = labels.reshape(-1,1)
        
        accuracy = calculate_accuracy(labels, labels0)
        
        accuracy0.append(accuracy)
        
        
    print(np.mean(accuracy0[:]))    
    print(nn_mising)
    nn_mising +=0.1
    # print(nn)
    # nn +=0.1
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    

  
    #%%

    # =============================================================================
    # Assume we know 95% of these labels
    known_labels = np.array([1] * 43 + [2] * (63 - 43))

    accuracy0 = []
    
    for i in range(10):
        labels = known_labels.copy()
        unknown_indices = np.random.choice(np.arange(63), size=int(63 * nn), replace=False)
        labels[unknown_indices] = -1  # Mark some samples as unknown
        
        # Function to find the nearest label for unknown samples
        def classify_unknown_samples(V, labels):
            for unknown_idx in unknown_indices:
                # Calculate distance from unknown sample to all known samples
                distances = []
                for i, label in enumerate(labels):
                    if label != -1:  # Only consider known labels
                        dist = custom_distance(V[unknown_idx], V[i])
                        distances.append((dist, label))
                
                # Sort by distance and take the label of the closest sample
                nearest_label = min(distances, key=lambda x: x[0])[1]
                labels[unknown_idx] = nearest_label  # Assign the nearest label to the unknown sample
        
            return labels
        
        # Classify unknown samples and get final labels
        final_labels = classify_unknown_samples(V, labels)
        # print("Final Labels:", final_labels)
        
        if len(V) ==63:
            labels = labels.reshape(63, 1)
  
        accuracy = calculate_accuracy(labels, labels0)
        
        print("Final accuracy:", accuracy)
        accuracy0.append(accuracy)
        
    print(np.mean(accuracy0[:]))
    
    
    
    
    
    

