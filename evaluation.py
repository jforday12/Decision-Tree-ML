"""
Functions for evaluating a decision tree 
"""
import numpy as np
import treebuilder
from copy import deepcopy
import decimal

def evaluate(test_dataset, trained_tree):
    """
    Evaluates decision tree using an unseen test dataset  

    Arguments:
        test_dataset -- dataset to test tree with
        trained_tree -- root node of tree to evaluate
    Returns:
        accuracy -- fraction of predictions tree got correct
        confusion matrix -- 4x4 vector
    """
    confusion_matrix = np.zeros((4, 4))
    correct_predictions = 0
    incorrect_predictions = 0 

    #for every row of test dataset find the label based on the trained decision tree 
    for row in test_dataset:
        # label determined by tree
        predicted_label = inference(row, trained_tree)
        # actual label
        actual_label = row[-1]
        # update confusion matrix based on if prediction is correct
        if predicted_label == actual_label:
            confusion_matrix[int(actual_label-1)][int(actual_label-1)] += 1
            correct_predictions += 1
        else:
            confusion_matrix[int(actual_label - 1)][int(predicted_label - 1)] += 1
            incorrect_predictions += 1
    # compute accuracy after evaluating test data set
    accuracy = ((correct_predictions) / (correct_predictions + incorrect_predictions)) 
    return accuracy, confusion_matrix

def inference(data, trained_tree):
    """
    Function to inference with decision tree using 7 WiFi signals
    to determine the room the user is in

    Arguments:
        data -- 7 WiFi signals to evaluate
        trained_tree -- decision tree 

    Returns:
        label -- room determined by tree
    """
    # return leaf value
    if trained_tree.left is None and trained_tree.right is None:
        return trained_tree.label
    else:
        # evaluate data point through tree 
        if data[trained_tree.split_feature] <= trained_tree.split_threshold:
            return inference(data, trained_tree.left)
        else:
            return inference(data, trained_tree.right)
        
def prune_cross_validation(dataset, seed):
    """
    Performs cross validation within cross validation across 
    10 fold splits using pruning to trim the decision tree to
    improve generalization and reduce overfitting.

    Arugments:
        dataset -- dataset to perform 10-fold cross val    
    Returns:
        prune_accuracy -- average test accuracy of the 10 best pruned
            trees from each fold
        prune_cm -- average cm using test set of the 10 best pruned 
            trees from each fold

    """
    # calculate metrics using test data (i)
    outer_folds = k_fold_split(dataset, 10, seed)
    # average accuracy and confusion matrix for best pruned trees
    prune_accuracy = 0
    tree_depth=0
    prune_cm = np.zeros((4,4))
    for fold in outer_folds:
        best_prune_accuracy = 0
        test_set = fold[1]
        inner_folds = k_fold_split(fold[0], 9, seed)
        for inner_fold in inner_folds:
            # take the test set of inner fold as validation set
            validation_set = inner_fold[1]
            training_set = inner_fold[0]
            # train tree with inner fold training set
            trained_tree, _ = treebuilder.decision_tree_learning(training_set)
            # deep copy tree for pruning 
            copied_trained_tree = deepcopy(trained_tree)
            # base accuracy of tree
            accuracy, _ = evaluate(validation_set, trained_tree)            
            # dfs prune copied tree, keeping prunes based on validation result
            treebuilder.dfs_prune(copied_trained_tree, copied_trained_tree, 
                                  validation_set)
            # accuracy of pruned tree
            pruned_acc, _ = evaluate(validation_set, copied_trained_tree)
            if(pruned_acc > best_prune_accuracy):
                best_prune_accuracy = pruned_acc
                best_pruned_tree = deepcopy(copied_trained_tree)
        # take accuracy and confusion matrix of the best tree 
        accuracy, cm = evaluate(test_set, best_pruned_tree)
        tree_depth+=get_tree_depth(best_pruned_tree)
        prune_accuracy += accuracy
        prune_cm += cm

    return prune_accuracy/10, prune_cm/10, tree_depth/10

def cross_validation(dataset, seed):
    """
    Performs cross validation on decision tree 
    
    Arguments:
        dataset -- dataset to use
    Returns:
        accuracy -- average accuracy
        confustion_matrix -- average confusion matri
        avg_depth -- the average depth of the trees generated
    """
    # create 10 folds of (testing, training) from dataset
    folds = k_fold_split(dataset, 10, seed)
    # average accuracy, depth and cm to be computed over the 10 folds
    avg_accuracy = 0
    avg_depth=0
    avg_confusion_matrix = np.zeros((4,4))
    for fold in folds:
        training_fold = fold[0]
        test_fold = fold[1]
        # compute decision tree with training fold
        decision_tree, depth = treebuilder.decision_tree_learning(training_fold)
        # evaluate decision tree using the testing data
        accuracy, confusion_matrix = evaluate(test_fold, decision_tree)
        # update accuracy, confusion matrix and depth
        avg_accuracy += accuracy
        avg_confusion_matrix += confusion_matrix
        avg_depth+=depth
    return avg_accuracy/10, avg_confusion_matrix/10, avg_depth/10

def k_fold_split(dataset, k, seed):
    """
    Creates k folds in dataset with k-1 for training  1 for testing 

    Arguments:
        dataset -- dataset to create folds
        k -- fold distribution
        seed -- for initial randomization
    Returns:
        k_folds --  k arrays of [training_fold, test_fold]
    """
    # shuffle dataset before folding
    np.random.seed(seed)
    np.random.shuffle(dataset)
    k_folds = []
    fold_size = len(dataset) // k
    folds = [] 
    for i in range(k):
        # split into k folds
        folds.append(dataset[i*fold_size : (i + 1) * fold_size])
    for i in range(k):
        # split test and training data from each fold
        test_fold = np.array(folds[i])
        test_fold = test_fold.reshape((fold_size, 8))
        training_fold = np.array(folds[:i] + folds[i+1:])
        training_fold = training_fold.reshape(((len(dataset) - fold_size), 8))
        k_folds.append([training_fold, test_fold])
    
    return k_folds

def get_precision(confusion_matrix, label):
    """
    Calculates the precision rate from the confusion matrix for a given class.  

    Arguments:
        confusion_martix -- The confusion matrix created from our algorithm
        label -- The class for which precision is to be calculated
    Returns:
        precision --  The precision of the algorithm with respect to a class.
    """
    tp = confusion_matrix[label-1][label-1]
    fp = 0
    for i in range(len(confusion_matrix)):
        # So it does not add the true positive to the false positives. 
        if i != label - 1:
            fp += confusion_matrix[i][label-1]
    precision = decimal.Decimal(tp) / decimal.Decimal(tp + fp)
    return precision

def get_recall(confusion_matrix, label):
    """
    Calculates the recall rate from the confusion matrix for a given class.  

    Arguments:
        confusion_martix -- The confusion matrix created from our algorithm
        label -- The class for which recall is to be calculated
    Returns:
        recall --  The recall of the algorithm with respect to a class.
    """
    tp = confusion_matrix[label-1][label-1]
    fn = 0
    for i in range(len(confusion_matrix)):
        # So it does not add the true positive to the false negatives
        if i != label -1:
            fn += confusion_matrix[label-1][i]
    recall = decimal.Decimal(tp) / decimal.Decimal(tp + fn)
    return recall

def print_metrics(confusion_matrix):
    """
    Calculate and print precision, recall and accuraccy for 
    each label for a given label (room) 
    """
    for i in range(1, 5):
        print("room ", i, "metrics: ")
        precision = get_precision(confusion_matrix, i)
        recall = get_recall(confusion_matrix, i)
        print("precision: ", precision)
        print("recall: ", recall)
        print("F1", getF1(recall, precision))
        print("\n")

def getF1(recall, precision):
    """
    Calculates the f1-measure derived from the recall and precision rates.

    Arguments:
        recall -- The recall rate.
        precision -- The precision rate
    Returns:
        f1 --  The F1-measure. 
    """
    # So we do not divide by 0
    if precision + recall == 0:
        return 0  
    f1 = decimal.Decimal((2 * (precision * recall))) / decimal.Decimal(precision + recall)
    return f1

def get_tree_depth(node):
    if node is None:
        return 0
    else:
        ldepth=get_tree_depth(node.left)
        rdepth=get_tree_depth(node.right)
    
        if (ldepth>rdepth):
            return ldepth+1
        else:
            return rdepth+1

    
    