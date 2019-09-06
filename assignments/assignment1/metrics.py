import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0


    for i in range(len(prediction)):
        if ground_truth[i] == True and prediction[i] == True:
            tp += 1
        elif ground_truth[i] == False and prediction[i] == False:
            #print ("TN += 1")
            #print ("prediction[i] = {}, ground_truth[i] = {}".format(prediction[i], ground_truth[i]))
            tn += 1
        elif ground_truth[i] == False and prediction[i] == True:
            fp += 1
        elif ground_truth[i] == True and prediction[i] == False:
            #print ("FN += 1")
            #print ("prediction[i] = {}, ground_truth[i] = {}".format(prediction[i], ground_truth[i]))
            fn += 1


    #print ("tn = {} , fn = {}".format(tn, fn))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp + tn)/(tp + tn + fp + fn)

    f1 = 2* (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/F1_score
    # https://en.wikipedia.org/wiki/Precision_and_recall

    #print ("accuracy = {}".format(accuracy))

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    counter = 0
    for i in range(len(prediction)):
        if ground_truth[i] == prediction[i]:
            counter += 1

    accuracy = counter / len(prediction)

    return accuracy
