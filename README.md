
#Code Description:
Given a dataset with 7 wifi signal strengths and associated room the signals were read in. This decision tree will learn from the dataset and return with up to 97% accuracy the location of the room given an unseen signal strength. 

The tree is then pruned to save on memory and time requirements. 
# Running the code:

Running `main.py` will run the cross validation without and with pruning for both datasets and print all the results (confusion matrix, accuracy, recall, F1 etc..).

Run time is approx 2 mins on lab machines.  

`evaluation.py` contains functions for evaluting the tree and `treebuilder.py` functions for building the tree and pruning. 

`cross_validation()` or `prune_cross_validation()` from `evaluation.py` (specifing a dataset and seed for each) are the main build a decision tree and evaluate and build a decision tree, prune and evaluate functions. Both can also be called from main via `custom_cross_validation()` and `custom_prune_cross_validation()` also specifiying a dataset and seed.
