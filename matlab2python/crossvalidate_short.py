import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import KFold
from scipy.stats import binom
from binomialTest import binomial_test
from classifyDoppler_short import classifyDoppler_short


def crossvalidate_short(data, labels, verbose=False, validationMethod='kFold', K=10, N=30,
                       classificationMethod='PCA+LDA', testData=None, m=1, variance_to_keep=95,
                       trial_ind=None):
    """
    crossValidate will load data from a set of session/run combinations
    and then cross validate those results using your method of choice
    
    Parameters:
    data: array of shape (nImagesL&R x nPixels*yPixels)
    labels: array of shape (nImages,) with values like [0, 1, 0, 0, 1, 1, ...]
    verbose: boolean - print results
    validationMethod: string - options are 'kFold'
    K: integer - for K-fold validation (default: 10)
    N: integer - parameter for validation (default: 30)
    classificationMethod: string - options are 'PCA+LDA'
    testData: array of same shape as data - optional test data from different time window
    m: positive integer - subspace dimensions for cPCA (max: # classes - 1, default: 1)
    variance_to_keep: float - variance to keep in PCA (default: 95)
    trial_ind: array - trial indices (default: all trials)
    
    Returns:
    cp: dictionary containing classification performance metrics
    p: p-value from binomial test
    class_tracker: array of predicted classes for each trial
    """
    
    # Handle default trial indices
    if trial_ind is None:
        trial_ind = np.arange(len(labels))
    
    # Normalize to z-score
    data = zscore(data, axis=0)  # standardize features
    if testData is not None:
        testData = zscore(testData, axis=0)
    
    # Get useful variables
    N_trials = data.shape[0]  # number of trials
    
    # Initialize class performance tracking
    cp = {
        'CorrectRate': 0.0,
        'CountingMatrix': np.zeros((len(np.unique(labels)), len(np.unique(labels)))),
        'ErrorRate': 0.0,
        'LastClassName': None,
        'ClassNames': np.unique(labels),
        'TestTargets': [],
        'TestResults': []
    }
    
    # k-fold cross validation
    if validationMethod == 'kFold':
        # Create k-fold indices
        kfold = KFold(n_splits=K, shuffle=True, random_state=42)
        class_tracker = np.full(N_trials, np.nan)
        
        fold_idx = 0
        # For each k-fold
        for train_idx, test_idx in kfold.split(data):
            fold_idx += 1
            
            # Get training and test data
            train_data = data[train_idx]
            train_labels = labels[train_idx]
            
            if testData is not None:
                test_data = testData[test_idx]
            else:
                test_data = data[test_idx]
            
            test_labels = labels[test_idx]
            
            # Classify using your classification function
            # Note: You need to implement classifyDoppler_short
            predicted_classes, model = classifyDoppler_short(
                train_data, train_labels, test_data,
                method=classificationMethod,
                m=m,
                variance_to_keep=variance_to_keep
            )
            
            # Update class performance
            classperf_update(cp, predicted_classes, test_labels)
            class_tracker[test_idx] = predicted_classes
            
            # Optional progress tracking
            # print(f'Finished fold {fold_idx}/{K}')
    
    # Calculate classification accuracy measures
    nCorrect = np.sum(np.diag(cp['CountingMatrix']))
    nCounted = np.sum(cp['CountingMatrix'])
    
    if nCounted > 0:
        cp['CorrectRate'] = nCorrect / nCounted
        cp['ErrorRate'] = 1 - cp['CorrectRate']
    
    percentCorrect = cp['CorrectRate'] * 100
    chance = 1 / len(np.unique(labels))
    
    # Binomial test for significance
    p = binomial_test(nCorrect, nCounted, chance, 'one')
    
    # Display measures if verbose is on
    if verbose:
        print(f'\nClassification Accuracy:')
        print(f'{int(nCorrect)} / {int(nCounted)} trials correctly classified ({percentCorrect:.2f}% correct)\t', end='')
        
        if p < 0.001:
            print('(binomial test: p < 0.001)')
        else:
            print(f'(binomial test: p = {p:.3f})')
    
    return cp, p, class_tracker


def classperf_update(cp, predicted_classes, true_labels):
    """
    Update classification performance metrics
    Equivalent to MATLAB's classperf function
    """
    # Convert to numpy arrays
    predicted_classes = np.array(predicted_classes).flatten()
    true_labels = np.array(true_labels)
    
    # Store test results
    cp['TestTargets'].extend(true_labels.tolist())
    cp['TestResults'].extend(predicted_classes.tolist())
    
    # Update confusion matrix
    unique_labels = cp['ClassNames']
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            mask_true = (true_labels == true_label)
            mask_pred = (predicted_classes == pred_label)
            cp['CountingMatrix'][i, j] += np.sum(mask_true & mask_pred)