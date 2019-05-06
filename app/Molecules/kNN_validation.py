

###Old KNN -  Adapt to the new task
### TODO
#   GED implementation
#   Change input data
#   Use GED distance instead of Euclidean distance



from collections import Counter
import csv
import math
import operator

# open and read training and test sets
Train_Labels = []
Train_Data = []
Test_Labels = []
Test_Data = []
with open('train.csv', 'r') as csvfile:
        lines = csv.reader(csvfile)
        for row in (lines):
            #print(row[0])
            Train_Labels.append(row[0])
            Train_Data.append(row[1:])
        #train_dataset = list(lines)
print('number of trainingset lines: ' + repr(len(Train_Data)))

with open('test.csv', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        # print(row[0])
        Test_Labels.append(row[0])
        Test_Data.append(row[1:])
#print((Test_Data[2]))
#print((Test_Data[1]))
print('number of testset lines: ' + repr(len(Test_Data)))


#Calculation of Euclidian distance
def EuclideanDistance(x,p,n):
    dist = 0
    for i in range (n):
        dist +=pow((int(x[i])-int(p[i])),2)
    return math.sqrt(dist)

#Calculation of Manhattan distance
def ManhattanDistance(x,p,n):
    dist=0
    for i in range (n):
        dist += abs(int(x[i])-int(p[i]))
    return dist

#Calcul all the neighbors et get the k closest
def Neighbors(TrainingSet, TestValue, k, distance):
    distances = []                                      # Table to store distance between test value (x) and training values (p)
    n=len(TestValue)
    neighbors = []                                      # Table to store the nearest neighbors
    for x in range(len(TrainingSet)):
        if distance == "Euclidean":
            dist=EuclideanDistance(TestValue, TrainingSet[x], n)
        elif distance == "Manhattan":
            dist = ManhattanDistance(TestValue, Train_Data[x], n)
        distances.append((Train_Labels[x], dist))
        # print('Test value: ' + repr(TestValue))
        # print('Train Value:' + repr(TrainingSet[x]))
        #print('distance ' + repr(x) +'=' + repr(distances[x]))
    distances.sort(key=operator.itemgetter(1))
    #sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
    #print('nb de distances calculées: ' + repr(len(distances)))
    #print("dist:" + repr(distances))# Sort all the distance

    for i in range(k):
        neighbors.append(distances[i][0])               # Store k neighbors
    return neighbors

#Predict/Assign the class
def Predict(neighbors):
    Counter(neighbors)
    PredictedClass = {}
    for x in range(len(neighbors)):
        value = neighbors[x]
        if value in PredictedClass:
            PredictedClass[value] += 1
        else:
            PredictedClass[value] = 1
    AssignedClass= max(PredictedClass.items(),key=operator.itemgetter(1))[0]
    return AssignedClass

#Calculate accuracy
def Accuracy(Prediction):
    correct = 0
    for i in range(len(Prediction)):
        if Test_Labels[i] is Prediction[i]:
            correct += 1
    accuracy = float(correct)/len(Prediction) *100  #accuracy
    return accuracy


#Main
TrainingSet = Train_Data
TestSet = Test_Data
AssignedClasses = []
#choose number of nearest neighbours and distance calculation
k=10
distance = "Euclidean" #"Manhattan" #

#for i in range(5):
for i in range(len(TestSet)):
    neighbors = Neighbors(TrainingSet, TestSet[i], k, distance)
    prediction = Predict(neighbors)
    AssignedClasses.append(prediction)
    #print('class neig' + repr(neighbors))
    print('step ' + repr(i) +': assigned =' + repr(AssignedClasses[i]) + ', real =' + repr(Test_Labels[i]))
accuracy = Accuracy(AssignedClasses)
print(repr(k) +'-NN Accuracy: ' + repr(accuracy) + '%' + ' with ' + distance + ' distance')




