#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# It is assigning data points to each cluster



def LABEL2(x,pt,eps,MinPoint,labels,cluster_value):
    NEIGHBORHOOD=[]
    LABEL_No=[]
    
    # It is calculating distance and comparing with epsilon 
    for i in range(x.shape[0]):
        if np.linalg.norm(x[pt]-x[i])<eps:
            NEIGHBORHOOD.append(x[i])
            LABEL_No.append(i)
    
    if len(NEIGHBORHOOD) < MinPoint:
        for i in range(len(labels)):
            if i in LABEL_No:
                labels[i]=-1
    else:
        for i in range(len(labels)):
            if i in LABEL_No:
                labels[i]=cluster_value
    
    return labels


#Plotting and count the Data Points in each cluster



def Mine_DBSCAN(data,eps,MinPoint):
    x=data
  
    labels=np.array([0]*x.shape[0])

    c=1

    for i in range(x.shape[0]):
        if(labels[i]==0):
            labels = LABEL2(x,i,eps,MinPoint,labels,c)        
    
    CLUSTER_NO,count = np.unique(labels,return_counts=True)
    
    
    
    #Datapoint Frequency in Each Cluster
    
    
    for i in range(len(CLUSTER_NO)):
        print(f"CLUSTER {i+1} has {count[i]} Data Points")
     
    
    
    #Plotting     
    plt.title(f" Clustering with DBSCAN for (eps={eps} , MinPoint={MinPoint})")
    plt.xlabel('RADIUS_MEAN')
    plt.ylabel('TEXTURE_MEAN')
    plt.scatter(x[:,0],x[:,1],c=labels,s=30,cmap='seismic')
    plt.show()
    
        
if __name__ == '__main__':
    data = pd.read_csv('cancer.csv')

    # Preprocessing Stage :Remove first Two Attributes
    df= data.iloc[:,2:32]   
    
    
    #Extracting only values 
    x_temp = df.values 
    
    # Preprocessing Stage : Data Transformation
    x = MinMaxScaler().fit_transform(x_temp) 
    
    #Executing the Algorithm by passing concerned parameters 
    
    print("CASE 1:")
    Mine_DBSCAN(x,0.2,6)
    print("CASE 2:")
    Mine_DBSCAN(x,0.5,6)
    print("CASE 3:")
    Mine_DBSCAN(x,0.2,3)
    
    


# In[ ]:




