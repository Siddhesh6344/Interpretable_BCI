import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import argparse

def classifyDoppler_short_code(trainData, trainLabels, testData, **kwargs):
    """
    Classifies doppler data using the training data and labels provided
    
    Parameters:
    trainData -- training data, shape (nImagesL&R, nPixels*yPixels)
    trainLabels -- training labels, shape (nImagesL&R,)
    testData -- testing data, shape (nImagesToTest, nPixels*yPixels)
    
    Keyword arguments:
    m -- scalar parameter (default: 1)
    method -- string of method type. Accepted values: 'PCA+LDA', 'LDA' (default: 'PCA+LDA')
    model -- pre-trained model dictionary (default: None)
    variance_to_keep -- for PCA dimensionality reduction (default: 95)
    
    Returns:
    pred -- predictions
    model -- trained model dictionary
    """
    
    # Default parameters
    m = kwargs.get('m', 1)
    method = kwargs.get('method', 'PCA+LDA')
    model = kwargs.get('model', None)
    variance_to_keep = kwargs.get('variance_to_keep', 95)
    
    # Remove NaN values
    nan_mask = np.isnan(trainLabels)
    trainLabels = trainLabels[~nan_mask]
    trainData = trainData[~nan_mask]
    
    # LDA Method
    if method == 'LDA':
        if model is None or not isinstance(model, dict):
            # Train new model
            trainPredictors = trainData
            MdlLinear = LinearDiscriminantAnalysis(solver='lsqr')  # pseudolinear equivalent
            MdlLinear.fit(trainPredictors, trainLabels)
            
            model = {
                'MdlLinear': MdlLinear
            }
        else:
            # Use existing model
            MdlLinear = model['MdlLinear']
        
        # Make predictions
        testDataPCA = testData
        pred = MdlLinear.predict(testDataPCA)
    
    return pred, model