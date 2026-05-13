




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats                        # common correlation
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import confusion_matrix
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent



nn_mising = 0.0

acc_ave = []

for i in range(5):
    
    #  datasets---: complext network without extra 20 houses
    V = pd.read_csv(BASE_DIR / "simulated_voltage_data_0_noise_1.csv") # need you put the excel or csv datasets in the same folders


    #  datasets---: complext network with extra 20 house
    #  datasets---: simple network longer cable
    #  datasets---: simple network 
    #  datasets---: complex network with three-phase meters
    #  datasets---: simple network with three-phase meters
    #  datasets---: complext network without extra 20 houses
    #  datasets---: complext network with extra 20 houses
    #  datasets---: simple network without extra 20 houses
    #  datasets---: simple network with extra 20 houses
    #  datasets---: complex network with three-phase meters
    #  datasets---: simple network with three-phase meters
    #  datasets---: complex network with three-phase meters--impact from PV decreases the accuracy
    # datasets---: complex network with three-phase meters
     
    
    V = np.array(V)
    V = V[1:,1:] # the scaler 230 does not bring enhancement
    # V = np.delete(V, np.random.choice(V.shape[1], int(V.shape[1] * nn_mising), replace=False), axis=1)
    
    
    
    # for i in range(len(V)):
    #     plt.plot(V[i,:])
    V = scaler.fit_transform(V)     # key point_ very important
    
    
    # ------------random missing part--------------------------
    # n = 0.0 # percentage of the missed data 
    # num_cols_to_delete = int(n * V.shape[1])
    # cols_to_delete = np.random.choice(V.shape[1], num_cols_to_delete, replace=False)
    # V = np.delete(V, cols_to_delete, axis=1)
    
    
    def calculate_accuracy(predicted_labels, true_labels):
      correct_predictions = np.sum(predicted_labels == true_labels)
      total_predictions = len(predicted_labels)
      accuracy = correct_predictions / total_predictions
      return accuracy
    
    #  lable of the houses at complex cables --------------------------------
    true_lables = [1,1,3,1,1,2,2,2,3,3,3,2,3,1,3,2,2,3,1,2,1,2,2,3,3,3,1,1] # one complex without 20
    
    
    
        # define the distance between two samples
    def custom_distance(x, y):
        a = 1
        pearson_corr, _ = stats.pearsonr(x, y) 
        pearson_corr = 1.1 - pearson_corr
        # pearson_corr = 1 - min(1/4 * math.log(1+np.exp(a+4*pearson_corr)),1) 
        return pearson_corr
    
    # pearson_corr = np.zeros((V.shape[0], V.shape[0]))
    # for i in range(len(V)):
    #     for j in range(len(V)):
    #         pearson_corr[i, j], _ = stats.pearsonr(V[i], V[j]) 
    #         # pearson_corr[i, j] = min(1/4 * math.log(1 + np.exp(1 + 4*pearson_corr[i, j])), 1) 
              
    
    linkage_matrix = linkage(V, method='complete', metric = custom_distance)  # same with the previous one
    # linkage_matrix = linkage(V, method='complete', metric = custom_distance)  # same with the previous one
    # linkage_matrix = linkage(pearson_corr, method='complete')  # same with the previous one
    labels = fcluster(linkage_matrix, 3, criterion='maxclust')
    print(labels)
    
    
    conf_matrix = confusion_matrix(labels, true_lables)
    print("conf_matrix:", conf_matrix)
    
    # Calculate the purity of each class----------------------
    n_classes = conf_matrix.shape[0]
    purity = np.zeros(n_classes)
    
    for i in range(n_classes):
        total_predictions = np.sum(conf_matrix[i, :])
        if total_predictions > 0:
            purity[i] = np.max(conf_matrix[i, :]) / total_predictions
        else:
            purity[i] = 0
    
    # print the purity of each class
    for i in range(n_classes):
        print(f"Class {i + 1} Purity: {purity[i]:.4f}")
    
    # Calculate the average purity of each class
    average_purity = np.sum(np.max(conf_matrix, axis=1)) / np.sum(conf_matrix)
    print(f"Average Purity: {average_purity:.4f}")
    
    
    accuracy = np.zeros(n_classes)
    
    for i in range(n_classes):
        true_positives = np.max(conf_matrix[:, i])
        total_actuals = np.sum(conf_matrix[:, i])
        if total_actuals > 0:
            accuracy[i] = true_positives / total_actuals
        else:
            accuracy[i] = 0
    
    
    # calculate the average accuracy
    average_accuracy = np.mean(accuracy)
    print(f"Average Accuracy: {average_accuracy:.4f}")
    
    acc_ave.append(average_accuracy)

print("***************************************")
acc_ave = np.mean(acc_ave)
print(f"Average Accuracy: {acc_ave:.4f}")


















