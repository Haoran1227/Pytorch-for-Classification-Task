from sklearn.metrics import f1_score
import numpy as np

def create_evaluation(label_list, pred_list, mode):
    """
    Args:
        label_list (list of tensor): true labels including only 0 and 1
        pred_list (list of tensor): predictions including only 0 and 1
        mode (string): 'train' or 'val'
    """

    if mode=='val':
        # convert tensor to ndarray
        pred_array = np.array([element.numpy() for element in pred_list]).reshape(-1, 2)
        label_array = np.array([element.numpy() for element in label_list]).reshape(-1, 2)
        crack_accuracy = f1_score(y_true=label_array[:, 0], y_pred=pred_array[:, 0], average='weighted')
        print('\ncrack_accuracy: ', crack_accuracy*100, '%')
        inactive_accuracy = f1_score(y_true=label_array[:, 1], y_pred=pred_array[:, 1], average='weighted')
        print('\ninactive_accuracy: ', inactive_accuracy*100, '%')
