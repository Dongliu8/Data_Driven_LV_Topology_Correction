

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:27:00 2025

@author: dliu8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats                        # common correlation
import scipy.signal
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import math
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import confusion_matrix

import networkx as nx
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw  # direcly calculate the DTW between two time series datasets, same with the defined one



nn_mising = 0.0

acc_ave = []

for i in range(5):

    
    # =============================================================================
    # successfully on 13/01/2025
    # =============================================================================
    
    #  datasets---: complext network without extra 20 houses
    V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\one_per_node/simulated_filtered_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\multiple_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\multiple_per_node/simulated_filtered_0.csv')

    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\one_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\one_per_node/simulated_filtered_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\multiple_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\multiple_per_node/simulated_filtered_29.csv')
    


    #  datasets---: complext network with extra 20 houses
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\one_per_node/simulated_filtered_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\multiple_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\multiple_per_node/simulated_filtered_0.csv')

    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\one_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\one_per_node/simulated_filtered_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\multiple_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\multiple_per_node/simulated_filtered_29.csv')
    
    
    #  datasets---: simple network longer cable
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\one_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\one_per_node/simulated_filtered_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\multiple_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\multiple_per_node/simulated_filtered_0.csv')


    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\multiple_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\multiple_per_node/simulated_filtered_29.csv')
    
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\one_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\one_per_node/simulated_filtered_29.csv')
    
    
    
    
    #  datasets---: simple network 
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\one_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\one_per_node/simulated_filtered_0.csv')

    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\multiple_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\multiple_per_node/simulated_filtered_0.csv')

    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\multiple_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\multiple_per_node/simulated_filtered_29.csv')
    
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\one_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\one_per_node/simulated_filtered_29.csv')
    
    
    
    
    
    #  datasets---: complex network with three-phase meters
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\one_per_node/simulated_filtered_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_filtered_0.csv')

    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\one_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\one_per_node/simulated_filtered_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_filtered_29.csv')
    
    
    #  datasets---: simple network with three-phase meters
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\one_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\multiple_per_node/simulated_0.csv')
    
    # wrong this is filtered not the unfiltered: V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\one_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\one_per_node/simulated_filtered_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\one_per_node/simulated_filtered_0.csv')
    
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\multiple_per_node/simulated_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\multiple_per_node/simulated_filtered_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\multiple_per_node/simulated_filtered_0.csv')
    
    
    
    
    
    # =============================================================================
    #  datasets
    # =============================================================================
    
    
    #  datasets---: complext network without extra 20 houses
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex\three_phases\multiple_per_node/simulated_voltage_data_rounded_0.csv')
    
    #  datasets---: complext network with extra 20 houses
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_more_connections\three_phases\multiple_per_node/simulated_voltage_data_rounded_0.csv')
    
    #  datasets---: simple network without extra 20 houses
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\one_per_node/simulated_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_longer_cable_lengths\three_phases\multiple_per_node/simulated_0.csv')
    
    #  datasets---: simple network with extra 20 houses
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable\three_phases\multiple_per_node/simulated_voltage_data_rounded_0.csv')
    
    #  datasets---: complex network with three-phase meters
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\one_per_node/simulated_voltage_data_rounded_0.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_0.csv')
    
    #  datasets---: simple network with three-phase meters
    
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\simple_network_one_cable_with_three_phase_meters\three_phases\multiple_per_node/simulated_0.csv')
    
    
    
    
    # =============================================================================
    # 
    # =============================================================================
    
    
    #  datasets---: complex network with three-phase meters--impact from PV decreases the accuracy
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_voltage_data_rounded_0 (1).csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_voltage_data_rounded_29 (1).csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_voltage_data_rounded_28.csv')
    
    
      # datasets---: complex network with three-phase meters
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\one_per_node/simulated_voltage_data_rounded_time_filtered_29.csv')
    # V = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_voltage_data_rounded_time_filtered_29.csv')
    
    
    
    # =============================================================================
    # successfully 
    # =============================================================================
    
    
    V = np.array(V)
    V = V[1:,1:] # the scaler 230 does not bring enhancement
    # V = np.delete(V, np.random.choice(V.shape[1], int(V.shape[1] * nn_mising), replace=False), axis=1)
    
    
    
    for i in range(len(V)):
        plt.plot(V[i,:])
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
    # true_lables = [1,1,3,1,1,2,2,2,3,3,3,2,3,1,3,2,2,3,1,2,1,2,2,3,3,3,1,1] # one complex without 20
    # true_lables = [1,1,3,1,1,2,2,2,3,3,2,3,3,2,3,2,2,1,1,3,2,2,3,1,2,1,2,1,2,2,1,2,1,2,1,3,1,2,3,3,3,1,1]
    # true_lables = [1,1,3,1,1,2,2,2,3,3,3,2,3,1,3,2,2,3,1,2,1,2,2,3,3,3,1,1,1,3,1,1,3,1,2,2,2,3,2,1,3,2,1,3,3,2,1,3]
    # true_lables = [1,1,3,1,1,2,2,2,3,3,2,3,3,2,3,2,2,1,1,3,2,2,3,1,2,1,2,1,2,2,1,2,1,2,1,3,1,2,3,3,3,1,1,1,3,1,1,3,1,2,2,2,3,2,1,3,2,1,3,3,2,1,3]
    
    #  lable of the houses at complex cables------------------------------------
    # true_lables = [1,2,3,1,1,1,1,3,3,3,2,2,1,2,1,3,3,2,1,3,1,2,1,2,1]
    # true_lables = [1,2,3,1,1,1,1,1,1,1,3,3,3,3,2,2,1,2,1,2,1,3,3,2,1,2,2,2,3,1,2,1,2,1]
    
    # true_lables =[3, 3, 1, 3, 3, 2, 2, 3, 2, 1, 1, 1, 2, 1, 1, 2, 
    #               1, 2, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 1, 3, 3, 2, 
    #               1, 3, 2, 2, 3, 2, 3, 2, 3, 1, 3 ,2 ,1 ,1 ,1 ,3 ,3]
    
    # true_lables =[1,1,3,1,1,2,2,1,2,3,3,3,3,2,3,1,3,2,2,3,1,1,2,3,1,1,2,3,2,3,3,3,1,1]	 # one_cable_complex_with_three_phase_meters
    # true_lables =[1,1,3,1,1,2,2,1,2,3,3,3,2,3,3,2,3,2,2,1,1,
    #               3,2,2,3,1,1,2,3,1,1,2,3,1,2,2,1,2,1,2,1,3,
    #               1,2,3,3,3,1,1]		
    
    # true_lables =[1,2,3,1,1,1,1,3,3,3,1,2,3,1,2,3,1,1,2,3,1,2,3,1,2,3,3,2,1,3,1,2,1,2,1] # 	simple_network_one_cable_with_three_phase_meters
    true_lables =[1,2,3,1,1,1,1,1,2,3,1,1,3,3,3,3,1,2,3,1,2,
                  3,1,2,1,1,2,3,1,2,3,1,2,3,3,2,1,1,2,3,2,2,
                  3,1,2,1,2,1]		
    
    
    
    
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
    
    
    # plt.figure(figsize=(10, 8))
    # plt.imshow(pearson_corr, cmap='jet', origin='lower', interpolation='nearest')
    # # plt.yticks(ticks=range(len(labels)), labels=labels, rotation=90, fontsize=8)
    # plt.colorbar(label='Pearson Correlation')
    # plt.title('Modified Pearson Correlation Matrix')
    # plt.show()
    
    # ---------------------------------------------------------
    
    conf_matrix = confusion_matrix(labels, true_lables)
    # print("conf_matrix:", conf_matrix)
    
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
    
    # print the accuracy of each class
    # for i in range(n_classes):
        # print(f"Class {i + 1} Accuracy: {accuracy[i]:.4f}")
    
    # calculate the average accuracy
    average_accuracy = np.mean(accuracy)
    print(f"Average Accuracy: {average_accuracy:.4f}")
    
    acc_ave.append(average_accuracy)

print("***************************************")
acc_ave = np.mean(acc_ave)
print(f"Average Accuracy: {acc_ave:.4f}")





# =============================================================================
# unsuccessfully
# =============================================================================

# V = np.array(V)
# V = V[1:,1:] # the scaler 230 does not bring enhancement
# for i in range(len(V)):
#     plt.plot(V[i,:])
# VV = scaler.fit_transform(V)     # key point_ very important


# # ------------random missing part--------------------------
# # n = 0.0 # percentage of the missed data 
# # num_cols_to_delete = int(n * V.shape[1])
# # cols_to_delete = np.random.choice(V.shape[1], num_cols_to_delete, replace=False)
# # V = np.delete(V, cols_to_delete, axis=1)



# def calculate_accuracy(predicted_labels, true_labels):
#   correct_predictions = np.sum(predicted_labels == true_labels)
#   total_predictions = len(predicted_labels)
#   accuracy = correct_predictions / total_predictions
#   return accuracy

# # true_lables = [1,1,3,1,1,2,2,2,3,3,3,2,3,1,3,2,2,3,1,2,1,2,2,3,3,3,1,1]
# # true_lables = [1,1,3,1,1,2,2,2,3,3,2,3,3,2,3,2,2,1,1,3,2,2,3,1,2,1,2,1,2,2,1,2,1,2,1,3,1,2,3,3,3,1,1]
# # true_lables = [1,1,3,1,1,2,2,2,3,3,3,2,3,1,3,2,2,3,1,2,1,2,2,3,3,3,1,1,1,3,1,1,3,1,2,2,2,3,2,1,3,2,1,3,3,2,1,3]
# # true_lables = [1,1,3,1,1,2,2,2,3,3,2,3,3,2,3,2,2,1,1,3,2,2,3,1,2,1,2,1,2,2,1,2,1,2,1,3,1,2,3,3,3,1,1,1,3,1,1,3,1,2,2,2,3,2,1,3,2,1,3,3,2,1,3]

# # true_lables = [1,2,3,1,1,1,1,3,3,3,2,2,1,2,1,3,3,2,1,3,1,2,1,2,1]
# # true_lables = [1,2,3,1,1,1,1,1,1,1,3,3,3,3,2,2,1,2,1,2,1,3,3,2,1,2,2,2,3,1,2,1,2,1]

# true_lables =[3, 3, 1, 3, 3, 2, 2, 3, 2, 1, 1, 1, 2, 1, 1, 2, 
#               1, 2, 2, 3, 3, 1, 2, 2, 1, 3, 3, 2, 1, 3, 3, 2, 
#               1, 3, 2, 2, 3, 2, 3, 2, 3, 1, 3 ,2 ,1 ,1 ,1 ,3 ,3]

#     # define the distance between two samples
# def custom_distance(x, y):
#     a = 1
#     pearson_corr, _ = stats.pearsonr(x, y) 
#     pearson_corr = 1 - min(1/4 * math.log(1+np.exp(a+4*pearson_corr)),1) 
#     return pearson_corr

# labelss = []
    
# for ii in range(3):
#     V = VV[:,ii*96:(ii+1)*96]
#     pearson_corr = np.zeros((V.shape[0], V.shape[0]))
#     for i in range(V.shape[0]):
#         for j in range(V.shape[0]):
#             pearson_corr[i, j], _ = stats.pearsonr(V[i], V[j]) 
#             pearson_corr[i, j] = min(1/4 * math.log(1 + np.exp(1 + 4*pearson_corr[i, j])), 1) 
              
    
#     # linkage_matrix = linkage(pearson_corr,method='complete', metric = dtw_distance) # the distance does not make sence
#     linkage_matrix = linkage(pearson_corr, method='complete', metric = custom_distance)
#     labels = fcluster(linkage_matrix, 3, criterion='maxclust')
#     # print("clustering labels:", labels)
#     labelss.append(labels)

# labelss = np.array(labelss).T

# label_final = np.zeros((labelss.shape[0], 3))
# for i, row in enumerate(labelss):
#     label_final[i, 0] = np.sum(row == 1)  
#     label_final[i, 1] = np.sum(row == 2)  
#     label_final[i, 2] = np.sum(row == 3)  

# binary_label_final = np.zeros_like(label_final)
# max_indices = np.argmax(label_final, axis=1)  
# for i, idx in enumerate(max_indices):
#     binary_label_final[i, idx] = 1  


# # plt.figure(figsize=(10, 8))
# # plt.imshow(pearson_corr, cmap='jet', origin='lower', interpolation='nearest')
# # # plt.yticks(ticks=range(len(labels)), labels=labels, rotation=90, fontsize=8)
# # plt.colorbar(label='Pearson Correlation')
# # plt.title('Modified Pearson Correlation Matrix')
# # plt.show()

# # ---------------------------------------------------------

# conf_matrix = confusion_matrix(max_indices+1, true_lables)
# print("conf_matrix:", conf_matrix)


# n_classes = conf_matrix.shape[0]
# purity = np.zeros(n_classes)

# for i in range(n_classes):
#     true_positives = conf_matrix[i, i]
#     total_predictions = np.sum(conf_matrix[i, :])
#     if total_predictions > 0:
#         purity[i] = true_positives / total_predictions
#     else:
#         purity[i] = 0

# for i in range(n_classes):
#     print(f"Class {i} Purity: {purity[i]:.4f}")







     


# =============================================================================
# unsuccessfully
# dbscan = DBSCAN(eps=0.08, min_samples=12).fit(V)
# labels = dbscan.labels_

# pca = PCA(n_components = 100)  # 降到30
# pca.fit(V)
# V = pca.transform(V)

# tsne = TSNE(n_components=3, random_state=0)
# V = tsne.fit_transform(V)

# kmeans = KMeans(n_clusters=3, random_state=0).fit(V)
# labels = kmeans.labels_

# list_to_remove = [10,11,12,13,14,15,17,18,19,20,21,22,23,24,25]
# V = np.delete(V, list_to_remove, axis=0)

# def custom_distance(series1, series2):
#     """Compute the DTW distance between two numeric sequences."""
#     # Convert to numpy arrays and ensure the dtype is float
#     series1, series2 = np.asarray(series1, dtype=float), np.asarray(series2, dtype=float)
#     n, m = len(series1), len(series2)
#     cost_matrix = cdist(series1[:, np.newaxis], series2[:, np.newaxis], metric='euclidean')
#     acc_cost_matrix = np.zeros((n + 1, m + 1))
#     acc_cost_matrix[0, 1:] = np.inf
#     acc_cost_matrix[1:, 0] = np.inf

#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             acc_cost_matrix[i, j] = cost_matrix[i - 1, j - 1] + min(
#                 acc_cost_matrix[i - 1, j],     # Insertion
#                 acc_cost_matrix[i, j - 1],     # Deletion
#                 acc_cost_matrix[i - 1, j - 1]  # Match
#             )
    # return acc_cost_matrix[n, m]
    
    
# V1 = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_voltage_data_rounded_29 (1).csv')
# V1 = np.array(V1)
# V1 = V1[1:,1:] # the scaler 230 does not bring enhancement
# V2 = pd.read_csv(r'C:\Users\dliu8\OneDrive - Delft University of Technology\Graduate School files\Conrerences and workshop\Alliander-06-12-2022\code\single_phase_qgis_imp\one_cable_complex_with_three_phase_meters\three_phases\multiple_per_node/simulated_voltage_data_rounded_28.csv')
# V2 = np.array(V2)
# V2 = V2[1:,1:] # the scaler 230 does not bring enhancement
# V = np.hstack((V1, V1))
# =============================================================================







