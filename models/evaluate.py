import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from data.dataset import CLASSES

def evaluate(preds, labels, classes=CLASSES):
    """
    Evaluates the model's predictions against the true labels and prints out accuracy, F1 scores, and a classification report.

    Args:
        preds:  numpy array of predicted labels
        labels: numpy array of true labels
        classes: list of class names
    """

    preds  = np.array(preds)
    labels = np.array(labels)
    
    present_classes = [classes[i] for i in sorted(np.unique(labels))]
    
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=present_classes))
    
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    return {'acc': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'cm': cm}