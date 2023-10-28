"""
Classes and functions for building, manipulating and 
visualizing a classification decision tree 
"""
import numpy as np
import evaluation
import matplotlib as plt

class Node:
    """
    Defines each Node in the decision tree
    """
    def __init__(self):
        """
        Constructs a Node
        """
        self.split_feature = None # feature to split with
        self.split_threshold = None # <= threshold to split with 
        self.left = None # left child
        self.right = None # right child
        self.label = None # classification label  
        self.majority_label = None # majority class at node
        self.x_pos = None # x coordinate for plotting node
        self.y_pos = None # y coordinate for plotting node 
        self.mod = 0 # property for shifting node's children
def decision_tree_learning(training_dataset, depth=0):
    """
    Recursive function to build the decision tree
    
    Arguments:
        training_dataset -- dataset used to build decision tree
        depth -- recursively tracks depth of tree (default 0)
    """
    # set of labels in dataset 
    unique_labels = set(training_dataset[:, -1])
    # if only one label then create leaf node as label
    if len(unique_labels) == 1:
        leaf = Node()
        leaf.label = training_dataset[0,-1]
        return leaf, depth 
    # multiple labels then create node for splitting 
    else:
        # find feauture and value to split by 
        split_feature, split_value = find_split(training_dataset)
        # create splitting node
        node = Node()
        node.split_feature = split_feature
        node.split_threshold = split_value
        # set new majority label
        node.majority_label = get_majority_label(training_dataset)
        #split the dataset based on new column and threshold 
        selected_column = training_dataset[:, split_feature]
        # create dataset for each side of criteria
        l_dataset = training_dataset[selected_column <= split_value]
        r_dataset = training_dataset[selected_column > split_value]
        #recursively run decision_tree_learning for each side
        l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
        node.left = l_branch
        node.right = r_branch
        return (node, max(l_depth, r_depth))

def get_majority_label(training_dataset):
        """
        Helper function to finds most popular label at a point in decision tree
        
        Arguments:
            training_dataset -- dataset to find most popular label 
        Returns:
            majority_label -- most popular label  
        """
        unique_labels = set(training_dataset[:, -1])
        maxlabel = 0
        for label in unique_labels:
            # count occurance of label in dataset
            label_count = len(training_dataset[training_dataset[:,-1] == label])
            # set new majority label if more than previous majority label
            if label_count >= maxlabel:
                maxlabel = label_count
                majority_label = label
        return majority_label

def find_split(dataset):
    """
    Helper function to find best splitting point (an attribute and threshold) for a node 
    in a decision tree based on which attribute and split threshold produces the
    highest information gain
    
    Arguments:
        dataset -- dataset to search for split point
    Returns:
        best_split_attribute -- best attribute for splitting
        best_threshold -- best threshold to split by 
    """
    best_information_gain = 0
    best_split_attribute = None
    best_threshold= None

    rows, columns = dataset.shape
    
    # loop through all attributes 
    for column in range(columns-1): 
        # sorted attributes values and remove duplicates
        sorted_values = sorted(set(dataset[:, column]))
        # loop through unique sorted attribute values
        for i in range(len(sorted_values)-1): 
            # threshold is midpoint two attributes 
            threshold = (sorted_values[i] + sorted_values[i+1]) / 2
            # split this attribute in dataset based on the threshold 
            selected_column = dataset[:, column]
            l_dataset = dataset[selected_column <= threshold]
            r_dataset = dataset[selected_column > threshold]
            # calculate information gain based on this split 
            information_gain = calc_information_gain(dataset, l_dataset, r_dataset)
            # set best information gain to calculated IG if better than previous
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_split_attribute = column
                best_threshold = threshold
    return best_split_attribute, best_threshold

def calc_information_gain(dataset, left_subtree, right_subtree):
    """
    Helper function to computer the information gain by splitting the dataset
    by a certain attribute and threshold 

    Arguments:
        dataset -- presplit dataset to calculate base entropy
        left_subtree -- left tree as a result of splitting
        right_subtree -- right tree as a result of splitting
    Returns: 
        information_gain -- calculated information gain

    """
    # entropy of presplit dataset
    dataset_entropy = calculate_entropy(dataset)
    # number of labels in the left and right subtree
    samples_left = len(left_subtree[:,-1])
    samples_right = len(right_subtree[:,-1])
    # total labels
    all_samples = samples_left + samples_right
    # left and right subtree entropy
    left_entropy = calculate_entropy(left_subtree)
    right_entropy = calculate_entropy(right_subtree)
    # calculate information gain
    remainder = ((samples_left / all_samples) * left_entropy) + ((samples_right / all_samples) * right_entropy)
    informationGain = dataset_entropy - remainder
    return informationGain         

def calculate_entropy(dataset):
    """
    Helper function to calculate the entropy of a dataset

    Arguments:
        dataset -- dataset to calculate entropy of
    Returns:
        entropy
    
    """
    entropy = 0
    rows, columns = dataset.shape
    unique_labels = set(dataset[:,-1])
    for label in unique_labels:
        # number of samples with the label k divided by the total number of samples from the initial dataset
        pk = len(dataset[dataset[:,-1] == label]) / rows
        entropy += pk * np.log2(pk)
    # entropy is negative
    return -entropy

def dfs_prune(root, current_node, validation_set):
    """
    Prunes the decision tree via a depth first search.

    Arguments: 
        root -- root node of decision tree,
        current_node -- node position in depth first search,
        validation_set -- dataset validate results of pruning      
    """
    # terminating condition - reached bottom of tree
    if current_node.left is None and current_node.right is None:
        return 
    # recusrively run deph first search
    dfs_prune(root, current_node.left, validation_set)
    dfs_prune(root, current_node.right, validation_set)
    # arrived at node with two leaf children
    if current_node.left.label is not None and current_node.right.label is not None:
        # determine accuracy without pruning 
        accuracy, _ = evaluation.evaluate(validation_set, root)
        # save leaves before pruning 
        left_save, right_save, = current_node.left, current_node.right
        # prune the node
        current_node.left = None
        current_node.right = None
        current_node.label = current_node.majority_label
        # determine accuracy with pruning
        pruned_accuracy, _ = evaluation.evaluate(validation_set, root)
        # if there is no improvement then revert prune
        if (pruned_accuracy < accuracy):
            current_node.left = left_save
            current_node.right = right_save
            current_node.label = None

"""
Functions for tree plotting, based on:

https://rachel53461.wordpress.com/2014/04/20/algorithm-for-drawing-trees/
"""


def draw_tree(node, plot):
    if not node:
        return
    if node.left:
        plot.plot([node.x_pos*10, node.left.x_pos*10], [-node.y_pos*2, -node.left.y_pos*2], 'k-')  # Draw line to left child
        draw_tree(node.left, plot)
    if node.right:
        plot.plot([node.x_pos*10, node.right.x_pos*10], [-node.y_pos*2, -node.right.y_pos*2], 'k-')  # Draw line to right child
        draw_tree(node.right, plot)
    # if node is leaf 
    if (node.label is not None):
        text = str(node.label)
    # node is threshold
    else:
        text = f"{node.split_feature} <= {node.split_threshold}"
    plot.text(node.x_pos*10, -node.y_pos*2, text, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))


def calculate_node_xy(root):
    calculate_inital_x(root)
    check_and_resolve_conflict(root.right, [root.left], 1)
    calculate_final_x(root)

def calculate_inital_x(node, depth=0, left_most=False):
    if node is None:
        return
    calculate_inital_x(node.left, depth+1, left_most=True)
    calculate_inital_x(node.right, depth+1)
    if left_most:
        node.x_pos = 0
    else:
        node.x_pos = 1
    node.y_pos = depth
    if(node.left is not None and node.right is not None):
        if left_most:
            node.x_pos = (node.left.x_pos + node.right.x_pos)/2
        else:
            node.x_mod = node.x_pos - (node.left.x_pos + node.right.x_pos)/2

def get_right_contour(node, depth, contour):
    if not node:
        return
    if depth not in contour:
        contour[depth] = node.x_pos
    else:
        contour[depth] = max(contour[depth], node.x_pos)
    get_right_contour(node.left, depth+1, contour)
    get_right_contour(node.right, depth+1, contour)

def get_left_contour(node, depth, contour):
    if not node:
        return
    if depth not in contour:
        contour[depth] = node.x_pos
    else:
        contour[depth] = min(contour[depth], node.x_pos)
    get_left_contour(node.left, depth+1, contour)
    get_left_contour(node.right, depth+1, contour)

def check_and_resolve_conflict(node, siblings, level):
    if not node or not siblings:
        return
    left_contour = {}
    get_left_contour(node, level, left_contour)
    shift = 0
    for sibling in siblings:
        right_contour = {}
        get_right_contour(sibling, level, right_contour)
        for y in left_contour:
            if (y in right_contour and left_contour[y] <= right_contour[y]):
                shift = max(shift, right_contour[y] - left_contour[y] + 1)
    node.x_pos += shift

def calculate_final_x(node, modsum=0):
    if node is None:
        return
    node.x_pos += modsum
    modsum += node.mod
    calculate_final_x(node.left, modsum)
    calculate_final_x(node.right, modsum)
    