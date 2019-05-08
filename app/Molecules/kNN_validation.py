import os
from collections import Counter
import csv
import math
import operator
import getData
import algorithm.graph_edit_dist as ged

path_train = 'C:/Users/Quentin.Meteier/Documents/Cours Uni/Pattern Recognition/Repo/PatternRecognition_JungleSpeed/data/MoleculesClassification/train.txt'
path_valid = 'C:/Users/Quentin.Meteier/Documents/Cours Uni/Pattern Recognition/Repo/PatternRecognition_JungleSpeed/data/MoleculesClassification/valid.txt'

labels_train = {}
labels_valid = {}

graphes_train = {}
graphes_valid = {}

def getLabels_train():
    with open(path_train, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            labels_train[int(row[0])] = row[1]
    return labels_train


def getLabels_valid():
    with open(path_valid, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            labels_valid[int(row[0])] = row[1]
    return labels_valid

def getGraphes_train():
    for key, value in graph_dict.items():
        if int(key) in labels_train:
            graphes_train[int(key)] = value

    return graphes_train

def getGraphes_valid():
    for key, value in graph_dict.items():
        if int(key) in labels_valid:
            graphes_valid[int(key)] = value

    return graphes_valid


#Calcul all the neighbors et get the k closest
def Neighbors(graphes_train, valid_label_key, k):

    # Store distances between given valid graph and training graphes
    distances = []
    for train_label_key in labels_train:
        # print(graphes_valid)
        # print(int(valid_label_key))
        # print(type(graphes_valid[int(valid_label_key)]))
        # print(graphes_train)
        # print(int(train_label_key))
        dist = ged.compare(graphes_valid[int(valid_label_key)], graphes_train[int(train_label_key)])
        distances.append((train_label_key, dist))

    distances.sort(key=operator.itemgetter(1))

    # Store k neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#Predict/Assign the class
def Predict(neighbors):
    PredictedClass = {}
    for id in neighbors:
        value = labels_train[id]
        if value in PredictedClass:
            PredictedClass[value] += 1
        else:
            PredictedClass[value] = 1
    #print(PredictedClass.items())
    assignedClass = max(PredictedClass.items(),key=operator.itemgetter(1))[0]
    return assignedClass

#Calculate accuracy
def Accuracy(prediction):
    correct = 0
    for i in range(len(prediction)):
        if labels_valid[prediction[i][0]] is prediction[i][1]:
            correct += 1
            print(correct)
        else:
            print("faux")
    accuracy = float(correct) / len(prediction) * 100  #accuracy
    return accuracy


def runKnn(graph_dict):

    AssignedClasses = []

    #choose number of nearest neighbours and distance calculation
    k = 3

    getLabels_train()
    getLabels_valid()

    getGraphes_train()
    getGraphes_valid()

    for i in labels_valid:
        neighbors = Neighbors(graphes_train, i, k)
        prediction = Predict(neighbors)
        AssignedClasses.append([i, prediction])

        # print(AssignedClasses[0][0])
        # print(labels_valid[AssignedClasses[0][0]])
        # print(i)
        # print(AssignedClasses[0][1])
        # print(len(prediction))
        # if labels_valid[AssignedClasses[0][0]] is AssignedClasses[0][1]:
        #     print("bravo")
        # else: print("c'est la merdre")
    #     #print('class neig' + repr(neighbors))
    #     print('step ' + repr(i) +': assigned =' + repr(AssignedClasses[i]) + ', real =' + repr(Test_Labels[i]))
    accuracy = Accuracy(AssignedClasses)
    print(repr(k) +'-NN Accuracy: ' + repr(accuracy) + '%')


###Old KNN -  Adapt to the new task
### TODO
#   GED implementation
#   Change input data
#   Use GED distance instead of Euclidean distance

if __name__ == "__main__":
    cwd = os.getcwd()
    graph_dict = getData.get_graphs()
    runKnn(graph_dict)




