import numpy as np
import evaluation
import treebuilder
import matplotlib.pyplot as plt

noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

# seed for custom experiments
SEED = 2525

def run_experiments(seed=123):
    """
    Runs all experiments and prints results 

    Arguments:
        seed -- seed for shuffling dataset 
    """
    print("Cross validation with no pruning experiments: \n\n")
    no_pruning_experiments(seed)
    print("-"*30,"\n")
    print("Cross validation within cross-validation pruning experiments:\n\n")
    pruning_experiments(seed)

def no_pruning_experiments(seed):
    """
    Runs cross validation experiments with no pruning
    
    Arguments:
        seed -- seed for shuffling dataset 
    """
    print("generating tree for clean dataset....")
    clean_acc, clean_cm, clean_depth = evaluation.cross_validation(clean_dataset, seed)
    print("avg depth = ", clean_depth, "\naccuracy = ", clean_acc, "\nconfusion matrix:\n", clean_cm,'\n')
    evaluation.print_metrics(clean_cm)
    print("generating tree for noisy dataset....")
    noisy_acc, noisy_cm, noisy_depth = evaluation.cross_validation(noisy_dataset, seed)
    print("avg depth = ", noisy_depth, "\naccuracy = ", noisy_acc, "\nconfusion matrix:\n", noisy_cm,'\n')
    evaluation.print_metrics(noisy_cm)

def pruning_experiments(seed):
    """
    Runs cross validation experiments with no pruning
    
    Arguments:
        seed -- seed for shuffling dataset 
    """
    print("generating tree for clean dataset....")
    clean_acc, clean_cm, clean_depth = evaluation.prune_cross_validation(clean_dataset, seed)
    print("avg depth = ", clean_depth, "\naccuracy = ", clean_acc, "\nconfusion matrix:\n", clean_cm,'\n')
    evaluation.print_metrics(clean_cm)
    print("generating tree for noisy dataset....")
    noisy_acc, noisy_cm, noisy_depth = evaluation.prune_cross_validation(noisy_dataset, seed)
    print("avg depth = ", noisy_depth, "\naccuracy = ", noisy_acc, "\nconfusion matrix:\n", noisy_cm,'\n')
    evaluation.print_metrics(noisy_cm)

def plot_graph():
    root, _ = treebuilder.decision_tree_learning(clean_dataset)
    treebuilder.calculate_node_xy(root)
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    treebuilder.draw_tree(root, ax)
    plt.show()

def custom_cross_validation(seed, raw_dataset):
    """
    Runs cross validation for a custom dataset 

    args: 
        seed: shuffling dataset pre k-fold split
        raw_dataset: txt file based dataset

    """
    print("generating tree for custom dataset....")
    acc, cm, depth = evaluation.cross_validation(raw_dataset, seed)
    print("avg depth = ", depth, "\naccuracy = ", acc, "\nconfusion matrix:\n", cm,'/n')
    evaluation.print_metrics(cm)

def custom_prune_cross_validation(seed, raw_dataset):
    """
    Runs pruned cross validation for a custom dataset 

    args: 
        seed: shuffling dataset pre k-fold split
        raw_dataset: txt file based dataset
    """
    print("generating tree for custom dataset....")
    acc, cm, depth = evaluation.prune_cross_validation(raw_dataset, seed)
    print("accuracy = ", acc, "\nconfusion matrix:\n", cm, "\navg depth = ", depth)
    evaluation.print_metrics(cm)


def main():
    """
    Runs predesined experiments, can uncomment custom experiments to run
    on unseen dataset 'raw_dataset'
    
    """
    run_experiments(SEED)

    #custom_cross_validation(SEED, raw_dataset=)
    #custom_prune_cross_validation(SEED, raw_dataset=)

if __name__ == "__main__":
    main()