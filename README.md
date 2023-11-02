# Running the code:

Following file structure is expected:
-----
./DecisionTrees
  -main.py
  -evaluation.py
  -treebuilder.py
  -/wifi_db
    -clean_dataset.txt
    -noisy_dataset.txt
-----

Running main.py will run the cross validation without and with pruning for both datasets and print all the results (confusion matrix, accuracy, F1 etc..) 

evaluation.py contains functions for evaluting the tree and treebuilder.py functions for building the tree and pruning. 

cross_validation() or prune_cross_validation() from evaluation.py (specifing a dataset and seed for each) are the main build a decision tree and evaluate and build a decision tree, prune and evaluate functions. Both can also be called from main via custom_cross_validation() and custom_prune_cross_validation() also specifiying a dataset and seed.
