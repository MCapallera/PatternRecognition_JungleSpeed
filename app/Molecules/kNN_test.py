import os
import csv
import operator
import getData
import algorithm.graph_edit_dist as ged

cwd = os.getcwd()
label_folder = cwd + '\\' + os.path.pardir + '\\' + os.path.pardir + '\\' + "data\MoleculesClassification\\"

labels_train = {}
ID_valid = []

graph_dict = {}
graphes_train = {}
graph_dict_test = {}

def getLabels_train():
    train_folder = os.path.join(label_folder, "train.txt")
    with open(train_folder, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            labels_train[int(row[0])] = row[1]
    return labels_train

def getID_valid():
    valid_folder = os.listdir(label_folder + "test\\gxl\\")
    #print(len(valid_folder))
    for file in valid_folder:
        ID_valid.append(os.path.splitext(file)[0])
    return ID_valid


def getGraphes_train():
    for key, value in graph_dict.items():
        if int(key) in labels_train:
            graphes_train[int(key)] = value
    #print(graphes_train)
    return graphes_train

# def getGraphes_valid():
#     for key, value in graph_dict.items():
#         if key in ID_valid:
#             graphes_valid[int(key)] = value
#
#     return graphes_valid


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
        dist = ged.compare(graph_dict_test[valid_label_key], graphes_train[int(train_label_key)])
        distances.append((train_label_key, dist))

    distances.sort(key=operator.itemgetter(1))

    # Store k neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#Predict/Assign the class
def predict(neighbors):
    predicted_class = {}
    for id in neighbors:
        value = labels_train[id]
        if value in predicted_class:
            predicted_class[value] += 1
        else:
            predicted_class[value] = 1
    #print(predicted_class.items())
    assignedClass = max(predicted_class.items(), key=operator.itemgetter(1))[0]
    return assignedClass

#Calculate accuracy
# def Accuracy(prediction):
#     correct = 0
#     for i in range(len(prediction)):
#         if labels_valid[prediction[i][0]] is prediction[i][1]:
#             correct += 1
#     accuracy = float(correct) / len(prediction) * 100  #accuracy
#     return accuracy

# Main function to run the kNN classification with GED
def runKnn():

    AssignedClasses = []

    #choose number of nearest neighbours
    k = 5

    getLabels_train()
    getID_valid()

    getGraphes_train()
    #getGraphes_valid()

    for i in ID_valid:
        print("Getting the " + str(k) + " neighbours for molcule n°" + str(i))
        neighbors = Neighbors(graphes_train, i, k)
        print(str(k) + "-nearest neighbours for molcule n°" + str(i) + " are : " + str(neighbors))
        prediction = predict(neighbors)
        print("Prediction for molcule n°" + str(i) + " is : " + str(prediction))
        AssignedClasses.append([i, prediction])
        exportResults(k, i, prediction)
        #print(AssignedClasses)
    #accuracy = Accuracy(AssignedClasses)
    #print(repr(k) + '-NN Accuracy: ' + repr(accuracy) + '%')

def exportResults(k, id, prediction):
    results_folder = cwd + '\\' + os.path.pardir + '\\' + os.path.pardir + '\\' + "results\MoleculesClassification\\"
    results_file = os.path.join(results_folder, "output_" + str(k) + "-NN_test.txt")
    if not os.path.isfile(results_file):
        file = open(results_file, 'w+')
        file.write(str(id) + "," + str(prediction) + "\n")
    else:
        with open(results_file, 'a') as file:
            file.write(str(id) + "," + str(prediction) + "\n")


if __name__ == "__main__":
    graph_dict = getData.get_graphs()
    graph_dict_test = getData.get_graphs_test()
    #print(graph_dict_test)
    runKnn()





