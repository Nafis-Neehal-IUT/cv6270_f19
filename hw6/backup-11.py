# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 03:53:36 2019

@author: nafis
"""

#------------------------------------------IMPORTS-----------------------------
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer


#------------------------------------------HELPER FUNCTIONS--------------------

"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)-1]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

"""
This function computes classwise performance of the classifier.
If dealing with training data, it will print the classwise training error.
If dealing with test data, it will print the classwise training accuracy.
"""
def calculate_classwise_performance(Y, Yp, label, 
                                    error=False):
    
    #indices where label matches with original data
    idxs = np.argwhere(Y==label)
    
    #indices where predicted data and original data has same values for 'idxs'
    idxs_t = np.argwhere(Yp[idxs]==label)
    
    if error:
        return (idxs.shape[0] - idxs_t.shape[0]) / idxs.shape[0]
    else:
        return idxs_t.shape[0] / idxs.shape[0]
    
"""
This function will do grid-search over a range of C values and max_iters
and return best parameters for each of K = 5 classes. 5 fold Cross Validation
has been introduced also for best estimation.
"""
def grid_search(X,Y):
    GSCVclfs = []
    parameters = {'C':[0.01,0.05,0.1,0.5,1,5,10]}
    numClasses = 5
    for k in range(numClasses):
        Y_copy = np.copy(Y)
        label = k+1
        Y_copy[Y_copy!=label] = -1
        Y_copy[Y_copy==label] = 1
        svc = LinearSVC()
        clf = GridSearchCV(svc, parameters, cv=5)
        clf.fit(X, Y_copy)
        GSCVclfs.append(clf)
    return GSCVclfs

"""
This function will generate 5 classfiers, each for one class with best 
parameter settings received from grid_search.
classes:[1,2,3,4,5] = C:[1,1,5,0.5,1]
"""
def classifier_generate(GSCVclfs, numClasses, X, Y):
    C_best = [GSCVclfs[i].best_params_['C'] for i in range(numClasses)]
    K_clfs = []
    for k in range(numClasses):
            Y_copy = np.copy(Y)
            label = k+1
            Y_copy[Y_copy!=label] = -1
            Y_copy[Y_copy==label] = 1
            clf = LinearSVC(C=C_best[k])
            clf.fit(X, Y_copy)
            K_clfs.append(clf)
    return K_clfs

"""
This function will 
- import the train/test data descriptor file
- seperate data and label vector
- randomize the indices
- transform (normalize) the data (X); label (Y) will be intact 
"""
def data_import_and_preprocess(FILE):
    
    #import data
    DataFile = FILE
    
    #Data = np.loadtxt(DataFile, delimiter='\t')
    
    
    with open(DataFile,'rb') as f:
        Data = pickle.load(f)
    
    #seperate data and label
    X = Data[...,:-1]
    Y = Data[...,-1]
    
    #shuffle the data indices - pseudorandom with seed 1
    row_indices_random = np.arange(X.shape[0])
    np.random.seed(1)
    np.random.shuffle(row_indices_random)
    
    #shuffled data
    X_shuffled = X[row_indices_random]
    Y_shuffled = Y[row_indices_random]

    #normalize the training data, no need for Labels
    transformer = Normalizer().fit(X_shuffled)
    X_shuffled_normalized = transformer.transform(X_shuffled)
    
    return X_shuffled_normalized, Y_shuffled

    
#--------------------------MAIN FUNCTION STARTS HERE---------------------------

if __name__== '__main__':
    
    #IMPORT the training dataset descriptor and Pre-Process (Normalize)
    trainFILE = 'trainDesc.pkl'
    trainX_shuffled_normalized, trainY_shuffled = data_import_and_preprocess(trainFILE)
    
    #number of classes
    numClasses = np.unique(trainY_shuffled).size
    
    
    #-----------------Classifier with best parameter generate------------------
    
    #grid search for best parameters (C)
    GSCVclfs = grid_search(trainX_shuffled_normalized, trainY_shuffled)
    
    #generate 5 classifiers with best parameter (C) settings for each K classes
    K_clfs = classifier_generate(GSCVclfs, numClasses, 
                                 trainX_shuffled_normalized, trainY_shuffled) 
    
    #--------------------Model Hyperparameters Calculate-----------------------
    
    #model hyperparameters
    W = np.array([K_clfs[i].coef_ for i in range(numClasses)])[:,-1,:]
    B = np.array([K_clfs[i].intercept_ for i in range(numClasses)])
    W_normalized = np.linalg.norm(W, axis=1)[...,np.newaxis]
        
    #---------------------Calculate Training Error-----------------------------
    
    #calculate training error
    trainX_shuffled_normalized_copy = trainX_shuffled_normalized.T
    D_train = (W@trainX_shuffled_normalized_copy + B)/W_normalized
    
    #predict training data labels using model hyperparameters
    trainY_predict = np.argmax(D_train, axis=0) + 1
    
    #calculate training error
    clf_train_error = [calculate_classwise_performance(trainY_shuffled, 
                                                       trainY_predict, i+1, 
                                                       error=True) 
                        for i in range(numClasses)]
    
    #--------------------Performance Evaluation on Test Data-------------------
    
    #import test data
    testFILE = 'testDesc.pkl'
    testX_shuffled_normalized, testY_shuffled = data_import_and_preprocess(testFILE)
    
    #transpose test data
    testX_shuffled_normalized = testX_shuffled_normalized.T
    
    #decision matrix for test data
    D_test = (W@testX_shuffled_normalized + B)/W_normalized
    
    #final prediction
    testY_predict = np.argmax(D_test, axis=0) + 1
    
    #test accuracy
    clf_model_accuracy = [calculate_classwise_performance(testY_shuffled, 
                                                          testY_predict, i) 
                            for i in range(1,6)]
    
    
    #---------------------------Model Evaluation------------------------------------
    
    # Print scores
    np.set_printoptions(precision=3)
    print("F1 Score:",f1_score(testY_shuffled, testY_predict, average='weighted'))
    print("Precision:",precision_score(testY_shuffled, testY_predict, average='weighted'))
    print("Recall:",recall_score(testY_shuffled, testY_predict, average='weighted'))
    print("Overall Accuracy:",accuracy_score(testY_shuffled, testY_predict))
    print("Classwise Training Error:",clf_train_error)
    print("Classwise Accuracy:",clf_model_accuracy)
    
    # Plot non-normalized confusion matrix
    class_names = np.array(['grass', 'ocean', 'redcarpet', 'road', 'wheatfield'])
    plot_confusion_matrix(testY_shuffled, testY_predict, classes=class_names,
                      title='Confusion matrix, without normalization')
    
    """
    Training Error: [0.18, 0.13375, 0.0075, 0.18875, 0.21125]
    Test Accuracy: [0.75, 0.765, 0.95, 0.76, 0.745]
    Confusin Matrix: 
        [[148   7   2   0  43]
         [  6 145   9   3  37]
         [  2   1 187   0  10]
         [ 22  20  20  20 118]
         [ 11   7   2   2 178]]
    F1 Score: 0.794
    Precision: 0.796
    Recall: 0.794
    Overall Accuracy: 0.794
    """
    
    
    
    
    
    
    
    
    
    
    
    