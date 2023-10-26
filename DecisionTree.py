import numpy as np
from copy import deepcopy

class Node:
    #Constructor for the Node Class
    def __init__(self):
        self.splitColumn = None
        self.splitThreshold = None
        self.left = None
        self.right = None
        self.label = None
        self.majorityLabel = None

#Main recursive function to build the decision tree
def decision_tree_learning(training_dataset, depth):
    #if all samples have the same label then 
    #set datastructure gets only unique labels
    uniqueLabels = set(training_dataset[:, -1])
    
    if len(uniqueLabels) == 1:
        #if there is only one unique label then all samples have the same label
        leaf = Node()
        #setting the leaf's label to that of the current dataset 
        leaf.label = training_dataset[0,-1]
        return leaf, depth 
    else:
        #gets the split value and column using the find split function
        split_column, split_value = find_split(training_dataset)
        node = Node()
        node.splitColumn = split_column
        node.splitThreshold = split_value

        #get the majority label for the dataset 

        maxLabel = 0
        for label in uniqueLabels:
            numberofLabels = len(training_dataset[training_dataset[:,-1] == label])
            if numberofLabels >= maxLabel:
                maxLabel = numberofLabels
                majorityClass = label
                
        node.majorityLabel = majorityClass

        #split the dataset based on the column and threshold 
        # right dataset is one that meets (column > value) criteria
        selectedColumn = training_dataset[:, split_column]
        l_dataset = training_dataset[selectedColumn <= split_value]
        r_dataset = training_dataset[selectedColumn > split_value]
        
        #recursively run decision_tree_learning for both sub-datasets
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
        node.left = l_branch
        node.right = r_branch
        return (node, max(l_depth, r_depth))

def find_split(dataset):
    bestIG = 0
    bestSplitColumn = None
    bestSplitValue = None

    rows, columns = dataset.shape

    for column in range(columns -1): #-1 to exclude the last column (labels)
        #sorting the value of the attribute
        #sorted set of each unique value for a given column
        sortedValues = sorted(set(dataset[:, column]))
        for i in range(len(sortedValues) -1): #for every value of a given column
            #split points that are between two examples in sorted order
            threshold = (sortedValues[i] + sortedValues[i+1]) /2
            selectedColumn = dataset[:, column]
            #temporarily split the dataset based on proposed attribute and threshold to find information gain
            l_dataset = dataset[selectedColumn <= threshold]
            r_dataset = dataset[selectedColumn > threshold]

            #Calculate the information gain using the suggested split
            infoGain = information_gain(dataset, l_dataset, r_dataset)

            #if the temporary split produces the highest information gain 
            if infoGain > bestIG:
                bestIG = infoGain
                bestSplitColumn = column
                bestSplitValue = threshold
    #after looping over every attribute and every split point, return them 
    return bestSplitColumn, bestSplitValue

def entropy(dataset):
    entropy = 0
    rows, columns = dataset.shape
    uniqueLabels = set(dataset[:,-1])

    for label in uniqueLabels:
        #pk is the number of samples with the label k divided by the total number of samples from the initial dataset
        pk = len(dataset[dataset[:,-1] == label]) / rows
        entropy += pk * np.log2(pk)
    return -entropy


def information_gain(dataset, leftSubTree, rightSubTree):
    #Equation for IG: gain(Sall, Sleft, Sright) = H(Sall) - Remainder(Sleft, Sright)
    datasetEntropy = entropy(dataset)
    #get the number of samples in the left and right subtree
    samplesLeft = len(leftSubTree[:,-1])
    samplesRight = len(rightSubTree[:,-1])
    allSamples = samplesLeft + samplesRight
    leftEntropy = entropy(leftSubTree)
    rightEntropy = entropy(rightSubTree)

    remainder = ((samplesLeft / allSamples) * leftEntropy) + ((samplesRight / allSamples) * rightEntropy)

    informationGain = datasetEntropy - remainder
    return informationGain         



def kFoldSplit(dataset, numberOfFolds):

    splits = []
    np.random.seed(123)
    np.random.shuffle(dataset)
    allFolds = []
    foldSize = len(dataset) // numberOfFolds
    
    folds = [] 
    for i in range(numberOfFolds):
        folds.append(dataset[i*foldSize : (i + 1) * foldSize])
    for i in range(numberOfFolds):
        
        testData = np.array(folds[i])
        testData = testData.reshape((foldSize, 8))

        trainingDataset = np.array(folds[:i] + folds[i+1:])

        trainingDataset = trainingDataset.reshape(((len(dataset) - foldSize), 8))
        allFolds.append([trainingDataset, testData])
    
    return allFolds

def pruneCrossValidation(dataset):

    # calculate metrics using test data (i)
    averagePrunedTree = 0
    averageUnPrunedTree = 0
    PrunedConfusionMatrix=np.zeros((4, 4))
    UnPrunedConfusionMatrix=np.zeros((4, 4))
    outerFolds = kFoldSplit(dataset, 10)
    for fold in outerFolds:
        bestAcc = 0
        bestUnprunedAcc = 0
        print("\n\STARTING NEW OUTER TREE\n\n")
        testData = fold[1]
        innerFolds = kFoldSplit(fold[0], 9)
        for innerFold in innerFolds:

            print("NEW INNER TREE")
            # loops over every inner fold and trains tree and completes pruning
            validationSet = innerFold[1]

            trainingSet = innerFold[0]
            
            trainedTree, _ = decision_tree_learning(trainingSet, 0)
            copiedTrainedTree = deepcopy(trainedTree)

            accuracy, _ = evaluate(validationSet, trainedTree)
            
            #print("Unpruned Accuracy For Inner Tree: ", accuracy)

            accuracytest, _ = evaluate(testData, trainedTree)


            print("Unpruned Accuracy For Inner Tree TESTTTTTT: ", accuracytest)


            print("Number of nodes in pruned tree: ", totalNodes(trainedTree))
            
            #postorder dfs to find potential leaf to prune
           
            dfsPrune(copiedTrainedTree, copiedTrainedTree, validationSet)

            prunedAcc, _ = evaluate(validationSet, copiedTrainedTree)
            if(prunedAcc > bestAcc):
                bestAcc = prunedAcc
                bestTree = deepcopy(copiedTrainedTree)

            # for unpruned
            if(accuracy > bestUnprunedAcc):
                bestUnprunedAcc = accuracy
                bestUnprunedTree = deepcopy(trainedTree)
            testPruneAccuracy, _ = evaluate(testData, copiedTrainedTree)


            print("pruned accuracy For Inner Tree: ", prunedAcc)
            print("Pruned Accuracy For Inner TreeTESTTTTT: ", testPruneAccuracy)
            print("Number of nodes in pruned tree: ", totalNodes(copiedTrainedTree))

        tempAccuracy, temp_pruned_ConfusionMatrix = evaluate(testData, bestTree)
        averagePrunedTree += tempAccuracy
        PrunedConfusionMatrix+=temp_pruned_ConfusionMatrix
        
        tempAccuracy, temp_unpruned_ConfusionMatrix  = evaluate(testData, bestUnprunedTree)
        averageUnPrunedTree += tempAccuracy
        UnPrunedConfusionMatrix += temp_unpruned_ConfusionMatrix
    print("\n\n\nPruned Average: ", averagePrunedTree / 10)
    print("\n\n\nPruned confusion matrix:\n", PrunedConfusionMatrix / 10)
    print("\n\n\nUnpruned Average: ", averageUnPrunedTree / 10)
    print("\n\n\nUnPruned confusion matrix:\n", UnPrunedConfusionMatrix / 10)

 

def dfsPrune(root, currentNode, validationSet):
    if currentNode.left is None and currentNode.right is None:
        return 

    #if to be pruned
    dfsPrune(root, currentNode.left, validationSet)
    dfsPrune(root, currentNode.right, validationSet)
    if currentNode.left.label is not None and currentNode.right.label is not None:
        # determine accuracy without pruning 
        oldAccuracy, _ = evaluate(validationSet, root)
        nodeL, nodeR, = deepcopy(currentNode.left), deepcopy(currentNode.right)
        #temp prune
        currentNode.left = None
        currentNode.right = None
        currentNode.label = currentNode.majorityLabel
        newAccuracy, _ = evaluate(validationSet, root)


        if (newAccuracy < oldAccuracy):
            #print(newAccuracy,oldAccuracy)
            currentNode.left = deepcopy(nodeL)
            currentNode.right = deepcopy(nodeR)
            currentNode.label = None


def totalNodes(root):
  # Base case
    if(root == None):
        return 0
 
    # Find the left height and the
    # right heights
    l = totalNodes(root.left)
    r = totalNodes(root.right)
 
    return 1 + l + r

def evaluate(test_db, trained_tree):
    #Takes a test dataset and a trained tree and returns the accuracy of the tree
    #first dimension predicted second is actual
    confusionMatrix = np.zeros((4, 4))
    corectPrediction = 0
    incorrectPrediction = 0 
    #for every row of test dataset find the label based on the trained decision tree 
    for row in test_db:
        #The Label produced by the tree
        suggestedLabel = parse(row, trained_tree)
        #the true label
        trueLabel = row[-1]
        if suggestedLabel == trueLabel:
            confusionMatrix[int(trueLabel-1)][int(trueLabel-1)] += 1
            corectPrediction += 1
        else:
            confusionMatrix[int(suggestedLabel - 1)][int(trueLabel - 1)] += 1
            incorrectPrediction += 1
    #accuracy is correct predictions divided by all predictions
    accuracy = ((corectPrediction) / (corectPrediction + incorrectPrediction)) 

    return accuracy, confusionMatrix

def parse(row, trained_tree):
    #recursive function that parses the tree to return the label based on a row from a dataset
    #check every possible outcome of node until the current node is a leaf 
    if trained_tree.left is None and trained_tree.right is None:
        #leaf node 
        return trained_tree.label
    else:
        #not a leaf node, therefore we compare both column and threshold to know which side of tree to parse 
        if row[trained_tree.splitColumn] <= trained_tree.splitThreshold:
            return parse(row, trained_tree.left)
        else:
            return parse(row, trained_tree.right)




#Step 1: Loading The Data
noisyDataset = np.loadtxt("./wifi_db/noisy_dataset.txt")
cleanDataset = np.loadtxt("./wifi_db/clean_dataset.txt")

#Passing in with the noisy and clean datasets both with initial depths of 0
noisyDecisionTree, noisyDepth = decision_tree_learning(noisyDataset, 0)
# cleanDecisionTree, cleanDepth = decision_tree_learning(cleanDataset, 0)
#visualize_tree(noisyTree)
#visualize_binary_tree(cleanDecisionTree)


# print("Clean Dataset:\n")
# pruneCrossValidation(cleanDataset)

print("\n\nNoisy Dataset:\n")
pruneCrossValidation(noisyDataset)

