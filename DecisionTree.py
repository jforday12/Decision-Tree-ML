import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import math
import treebuilder

def kFoldSplit(dataset, numberOfFolds):

    splits = []
    # np.random.seed(123)
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
            
            print("Unpruned Accuracy For Inner Tree: ", accuracy)

            accuracytest, _ = evaluate(testData, trainedTree)
            #averageUnPrunedTree += accuracytest

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
        tempAccuracy, _ = evaluate(testData, bestTree)
        averagePrunedTree += tempAccuracy

        tempAccuracy, _ = evaluate(testData, bestUnprunedTree)
        averageUnPrunedTree += tempAccuracy
    print("\n\n\nPruned Average: ", averagePrunedTree / 10)
    print("\n\n\nUnpruned Average: ", averageUnPrunedTree / 10)

 

def dfsPrune(root, currentNode, validationSet):
    if currentNode.left is None and currentNode.right is None:
        return 

    #if to be pruned
    dfsPrune(root, currentNode.left, validationSet)
    dfsPrune(root, currentNode.right, validationSet)
    if currentNode.left.label is not None and currentNode.right.label is not None:
        # determine accuracy without pruning 
        oldAccuracy, _ = evaluate(validationSet, root)
        nodeL, nodeR, = currentNode.left, currentNode.right
        #temp prune
        currentNode.left = None
        currentNode.right = None
        currentNode.label = currentNode.majorityLabel
        newAccuracy, _ = evaluate(validationSet, root)

        
        if (newAccuracy < oldAccuracy):
            currentNode.left = nodeL
            currentNode.right = nodeR
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

def visualize_tree(node, x_positions, y_positions):
    """
    Function which visualizes the deecision tree
    
    """
    if node is not None:
        node_text = ""
        # find if node is leaf or threshold node:
        if node.label is not None:
            node_text = str(node.label)
            plt.text(x_positions[node], -y_positions[node], node_text, ha='center', bbox=dict(boxstyle="square", color = 'g'))
        else:
            node_text = f"{node.splitColumn} <= {node.splitThreshold}"
            plt.text(x_positions[node], -y_positions[node], node_text, ha='center', bbox=dict(boxstyle="square", color = 'y'))

        # plot the contents of node
        if node.left:
            # draws line from node to child left node 
            plt.plot([x_positions[node], x_positions[node.left]], [-y_positions[node], -y_positions[node.left]], 'b-')
            visualize_tree(node.left, x_positions, y_positions)
        if node.right:
            # draws line from parent to child right node 
            plt.plot([x_positions[node], x_positions[node.right]], [-y_positions[node], -y_positions[node.right]], 'b-')
            visualize_tree(node.right, x_positions, y_positions)

def get_positions(node, depth, x_position, y_positions, x_positions, spacing=10):
    if node is not None:
        y_positions[node] = -depth  # Negative because we want the root at the top
        x_positions[node] = x_position

        # Recursively assign positions for left and right children
        get_positions(node.left, depth + 1, x_position - spacing / (depth + 2), y_positions, x_positions)
        get_positions(node.right, depth + 1, x_position + spacing / (depth + 2), y_positions, x_positions)

# Function to print level order traversal of tree
 
#Step 1: Loading The Data
noisyDataset = np.loadtxt("./wifi_db/noisy_dataset.txt")
cleanDataset = np.loadtxt("./wifi_db/clean_dataset.txt")


tree, depth = decision_tree_learning(cleanDataset, 0)
noisyTree, _ = decision_tree_learning(noisyDataset, 0)

""" Plotting Functions """
x_positions = {}
y_positions = {}
get_positions(tree, 0, 0, x_positions, y_positions)
#get_log_positions(tree, 0, x_positions, y_positions)
fig, ax = plt.subplots()
visualize_tree(tree, x_positions, y_positions)
plt.show()


""" PRUNING FUNCTIONS """
# print("Clean Dataset:\n")
# pruneCrossValidation(cleanDataset)

#print("\n\nNoisy Dataset:\n")
#pruneCrossValidation(noisyDataset)

