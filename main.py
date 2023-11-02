import numpy as np
import evaluation
import treebuilder
import matplotlib.pyplot as plt

noisy_dataset = np.loadtxt("./wifi_db/noisy_dataset.txt")
clean_dataset = np.loadtxt("./wifi_db/clean_dataset.txt")

def run_experiments():
    """
    Runs cross_validation experiments and prints results
    
    """
    clean_acc, clean_cm = evaluation.cross_validation(clean_dataset)
    noisy_acc, noisy_cm = evaluation.cross_validation(noisy_dataset)
    p_clean_acc, p_clean_cm = evaluation.prune_cross_validation(clean_dataset)
    p_noisy_acc, p_noisy_cm = evaluation.prune_cross_validation(noisy_dataset)

    print("CLEAN DATASET PERFORMANCE: \n NO PRUNING")      
    print("ACCURACY = ", clean_acc)
    print("CONFUSION MATRIX = \n", clean_cm)
    for i in range(1,5):
        precision = evaluation.getPrecision(clean_cm, i)
        recall = evaluation.getRecall(clean_cm, i)
        print("Room ",i," precision: ", precision)
        print("Room ",i," recall: ", recall)
        print("Room ",i," F1-Measure", evaluation.getF1(recall, precision))
    print("PRUNING")   
    print("ACCURACY = ", p_clean_acc)
    print("CONFUSION MATRIX = \n", p_clean_cm)
    for i in range(1,5):
        precision = evaluation.getPrecision(p_clean_cm, i)
        recall = evaluation.getRecall(p_clean_cm, i)
        print("Room ",i," precision: ", precision)
        print("Room ",i," recall: ", recall)
        print("Room ",i," F1-Measure", evaluation.getF1(recall, precision))
    print("NOISY DATASET PERFORMANCE: \n NO PRUNING")   
    print("ACCURACY = ", noisy_acc)
    print("CONFUSION MATRIX = \n", noisy_cm)
    for i in range(1,5):
        precision = evaluation.getPrecision(noisy_cm, i)
        recall = evaluation.getRecall(noisy_cm, i)
        print("Room ",i," precision: ", precision)
        print("Room ",i," recall: ", recall)
        print("Room ",i," F1-Measure", evaluation.getF1(recall, precision))
    print("PRUNING")   
    print("ACCURACY = ", p_noisy_acc)
    print("CONFUSION MATRIX = \n", p_noisy_cm)
    for i in range(1,5):
        precision = evaluation.getPrecision(p_noisy_cm, i)
        recall = evaluation.getRecall(p_noisy_cm, i)
        print("Room ",i," precision: ", precision)
        print("Room ",i," recall: ", recall)
        print("Room ",i," F1-Measure", evaluation.getF1(recall, precision))


def plot_graph():
    root, _ = treebuilder.decision_tree_learning(clean_dataset)
    plt.figure(figsize=(30, 20))
    max_h_spacing = 20.0  # Adjust the maximum horizontal spacing as needed
    treebuilder.plot_decision_tree(root, x_center=0.5, y=1.0, max_h_spacing=max_h_spacing)

    plt.axis('off')
    plt.show()

def main():
    run_experiments()
    plot_graph()
    

if __name__ == "__main__":
    main()