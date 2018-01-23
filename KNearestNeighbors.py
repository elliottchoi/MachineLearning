from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import random
import pickle

style.use('fivethirtyeight')

# plot1 = [1,3]
# plot2=[2,5]
# 
# euclidean_distance = sqrt((plot1[0]-plot2[0])**2+(plot1[1]-plot2[1])**2)
# 
# print euclidean_distance

#r class with 3 vectors and r class with 3 vectors (my establihsed labeled k classes
dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
#The data feature we will use k means to classify
new_features=[5,7]

# #interate through class k and r
# for i in dataset:
#     #itterate through individual vectors in each class
#     for ii in dataset[i]:
#         #plot the 2-D vector, and assign colour for each class (ie k,r)
#         plt.scatter(ii[0],ii[1],s=100,color=i)
#         
#alternate one line loop
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]]for i in dataset]

#show new feature
plt.scatter(new_features[0], new_features[1], s=100)
plt.show()

def k_nearest_neightbors(data,predict,k=3):
    if len(data)>=k:
        #k represents the vote weighting we are using
        warnings.warn ('K is set to a value less than total voting groups!')
    distances=[]
    #analyze each class in the data
    for group in data:
        #itterate through each x,y,z...n point
        for features in data[group]:
            #sum the array -predict features, does pairwise linear algebra
            #euclidean_distances=np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            euclidean_distances=np.linalg.norm(np.array(features)-np.array(predict))
            #append to the distance, the group (so which class) and the euclidean distance
            distances.append([euclidean_distances,group])
    #sort the distances array from smallest to largest euclidean distance, and record which group it is from 
    votes =[i[1] for i in sorted(distances)[:k]]
    #print (distances)
    #print (sorted(distances))
    #print (Counter(votes).most_common(1))
    vote_result=Counter(votes).most_common(1)[0][0]
    #Establishing confidence, puts most common and how many were the msot common divided by the k (voting power)
    confidence=(Counter(votes).most_common(1)[0][1]/k)
    #print confidence
    return vote_result,confidence

#result=k_nearest_neightbors(dataset, new_features, k=3)
#print(result)

# [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# #assignment of new point via k means clustering 
# plt.scatter(new_features[0],new_features[1],s=100,color=result)
# plt.show()

#analyzing past data
accuracies=[]
for i in range(5):
    df=pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace=True)
    
    #data has been converted to a list of lists (each vector is all the attributes)
    full_data=df.astype(float).values.tolist()
    random.shuffle(full_data)
    # print(full_data[:5])
    # random.shuffle(full_data)
    # print(20*'#')
    # print(full_data[:5])
    
    test_size=0.4
    #two stands for benign, 4 stands for malignant
    train_set={2:[],4:[]}
    test_set={2:[],4:[]}
    #full data to the negative integer value of test size*test size of full data, slice by index value
    #this would be up to the last 20% of data
    train_data=full_data[:-int(test_size*len(full_data))]
    #last 20% of the data
    test_data=full_data[-int(test_size*len(full_data)):]
    
    #populate the dictionary
    for i in train_data:
        #-1 first element in the list (last value in the column, so 2 for benign 4 for malignent)
        #i[-1] will return 2 or 4
        train_set[i[-1]].append(i[:-1])
    
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    
    #pass the information to K nearest numbers
    correct=0
    total=0  
    
    for group in test_set:
        for data in test_set[group]:
            vote,confidence=k_nearest_neightbors(train_set, data, k=5)
            if group==vote:
                correct+=1
            total+=1
    accuracy=float(correct/total)
    #print 'Correct: ',correct,' Total: ',total, 'Accuracy',accuracy
    accuracies.append(correct/total)
    
print (sum(accuracies)/len(accuracies))

#comparison
pickler_in=open('KNearestNeighbors.pickle','rb')
clf=pickle.load()


