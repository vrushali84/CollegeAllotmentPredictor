import pandas as pd
import numpy as np
import csv
import math
from sklearn import neighbors, datasets
from numpy.random import permutation
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from pandas import DataFrame

data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
processed_data = data[['kcet','college']]
random_indices = permutation(data.index)
test_cutoff = math.floor(len(data)/5)
test = processed_data.loc[random_indices[1:test_cutoff]]
train = processed_data.loc[random_indices[test_cutoff:]]
train_output_data = train['college']
train_input_data = train
train_input_data = train_input_data.drop('college',1)
test_output_data = test['college']
test_input_data = test
test_input_data = test_input_data.drop('college',1)






class algorithms():
    
    def euclideanDistance(self,data1, data2, length):
        distance = 0
        for x in range(length):
            distance += np.square(data1[x] - data2[x])
        return np.sqrt(distance)


    def knn(self,trainingSet, testInstance, k):
        print(k)
        distances = {}
        sort = {}
        length = testInstance.shape[1]
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet.iloc[x], length)
            distances[x] = dist[0]
        sorted_d = sorted(distances.items(), key=lambda x: x[1])
        neighbors = []
        for x in range(k):
            neighbors.append(sorted_d[x][0])
        classVotes = {}
        for x in range(len(neighbors)):
            response = trainingSet.iloc[neighbors[x]][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return (sortedVotes, neighbors)
    
    def getAccuracy(self,testSet,predictions):
        correct=0
        for x in range(len(testSet)):
            if testSet[x][-1] is predictions[x]:
                correct+=1
        return (correct/float(len(testSet)))*100.0
    
                    
    def predictSVM(self,kcet,caste):
        
        if caste=="GOBC":
            print("caste obc")
            data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
            processed_data = data[['kcet','college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data)/5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college',1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',1)
            user_caste=str(caste)
            clf =svm.SVC()    
            clf.fit(train_input_data,train_output_data)
            marks=float(kcet)
            output_college1=clf.predict([[marks]])
            output_college=output_college1[0]
            return output_college
        
        elif caste=="GOPEN":
            print("caste open")
            data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
            processed_data = data[['kcet','college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data)/5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college',1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',1)
            user_caste=str(caste)
            clf =svm.SVC()
            clf.fit(train_input_data,train_output_data)
            marks=float(kcet)
            output_college1=clf.predict([[marks]])
            output_college=output_college1[0]
            return output_college
    
    def predictKNN(self,kcet,caste):
        user_caste=str(caste)
        if caste=="GOBC":
            data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
            processed_data = data[['kcet','college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data)/5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college',1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',1)

        elif caste=="GOPEN":
            data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
            processed_data = data[['kcet','college']]
            random_indices = permutation(data.index)
            test_cutoff = math.floor(len(data)/5)
            test = processed_data.loc[random_indices[1:test_cutoff]]
            train = processed_data.loc[random_indices[test_cutoff:]]
            train_output_data = train['college']
            train_input_data = train
            train_input_data = train_input_data.drop('college',1)
            test_output_data = test['college']
            test_input_data = test
            test_input_data = test_input_data.drop('college',1)
    
        kcet = float(kcet)
        testSet = [[kcet]]
        test = pd.DataFrame(testSet)
        k = 5
        result, neigh = self.knn(processed_data, test, k)
        list1 = []
        list2 = []
        for i in result:
            list1.append(i[0])
        for i in result:
            list2.append(i[1])
        for i in list1:
            print(i)
        return list1
    
    def top_ten(self):
        data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
        my_list1=data.nlargest(10, ['kcet'])
        my_list=my_list1.values.tolist()
        return my_list
    
    def get_by_range(self,start,end,results,caste):
        res=results
        if caste=="GOPEN":
            data = pd.read_csv('cutofflist.csv',encoding= 'unicode_escape')
        elif caste=="GOBC":
            data = pd.read_csv('cutoffobc.csv',encoding= 'unicode_escape')
        
        my_data=data = data[(data['kcet'] >= start) & (data['kcet'] <= end)]
        l1=data.head(res)
        my_list=l1.values.tolist()  
        return my_list

c=algorithms()
testSet=[[99.964,'College of Engineering, Pune'],[99.164,'College of Engineering, Pune'],[99.964,'Pune Institute of Computer Technology, Dhankavdi, Pune'],[99.409,'Walchand College of Engineering, Sangli']]   
predictions=['College of Engineering, Pune','Pune Institute of Computer Technology, Dhankavdi, Pune','Pune Institute of Computer Technology, Dhankavdi, Pune','Walchand College of Engineering, Sangli']
accuracy=c.getAccuracy(testSet,predictions)
print("--------------------Accuracy--------")
print("Accuracy of knn is {}".format(accuracy))




