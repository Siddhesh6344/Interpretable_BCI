import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def classifyDoppler_short(train_data, train_labels, test_data, **kwargs):
    """
    Classifies doppler data using the training data and labels provided
    
    Parameters:
    -----------
    train_data : array_like
        Training data, size (nImagesL&R, nPixels*yPixels)
    train_labels : array_like
        Training labels, size (nImagesL&R,)
    test_data : array_like
        Testing data of size (nImagesToTest, nPixels*yPixels)
    
    Keyword Arguments:
    ------------------
    m : int, optional
        Scalar parameter (default: 1)
    method : str, optional
        Method type. Accepted values: 'PCA+LDA' (default: 'PCA+LDA')
    model : dict or None, optional
        Pre-trained model (default: None)
    variance_to_keep : float, optional
        Percentage of variance to keep for PCA (default: 95)
        
    Returns:
    --------
    pred : ndarray
        Predictions for test data
    model : dict
        Trained model containing PCA and LDA components
    """
    # print("train_data shape:", train_data.shape)
    # print("train_data dtype:", train_data.dtype)
    # print("Contains NaN in train_data:", np.isnan(train_data).any())
    
    # Parse keyword arguments
    m = kwargs.get('m', 1)
    method = kwargs.get('method', 'PCA+LDA')
    model = kwargs.get('model', None)
    variance_to_keep = kwargs.get('variance_to_keep', 95)
    
    # Convert to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    
    # Remove NaN values if appropriate
    nan_indices = np.isnan(train_labels)
    train_labels = train_labels[~nan_indices]
    train_data = train_data[~nan_indices]
    
    # PCA+LDA method
    if method == 'PCA+LDA':
        if model is None or not isinstance(model, dict):
            # Train new model
            
            # PCA
            pca = PCA()
            pca_scores = pca.fit_transform(train_data)
            pca_coefficients = pca.components_.T  # Transpose to match MATLAB convention
            explained_variance_ratio = pca.explained_variance_ratio_
            pca_centers = pca.mean_
            
            # Determine number of components to keep
            explained_variance_to_keep_as_fraction = variance_to_keep / 100
            
            if variance_to_keep < 100:
                cumsum_explained = np.cumsum(explained_variance_ratio)
                num_components_to_keep = np.where(cumsum_explained >= explained_variance_to_keep_as_fraction)[0][0] + 1
                pca_coefficients = pca_coefficients[:, :num_components_to_keep]
                train_predictors = pca_scores[:, :num_components_to_keep]
            else:
                train_predictors = pca_scores
            
            # LDA
            mdl_linear = LinearDiscriminantAnalysis()
            mdl_linear.fit(train_predictors, train_labels)
            
            # Create model structure
            model = {
                'pca_centers': pca_centers,
                'pca_coefficients': pca_coefficients,
                'mdl_linear': mdl_linear
            }
        else:
            # Use existing model
            pca_centers = model['pca_centers']
            pca_coefficients = model['pca_coefficients']
            mdl_linear = model['mdl_linear']
        
        # Transform test data using PCA
        test_data_pca = (test_data - pca_centers) @ pca_coefficients
        
        # Make predictions
        pred = mdl_linear.predict(test_data_pca)
    
    else:
        raise ValueError(f"Unknown method: {method}. Only 'PCA+LDA' is supported.")
    
    return pred, model

# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.impute import SimpleImputer

# def classifyDoppler_short(train_data, train_labels, test_data, **kwargs):
#     """
#     Classifies doppler data using the training data and labels provided
    
#     Parameters:
#     -----------
#     train_data : array_like
#         Training data, size (nImagesL&R, nPixels*yPixels)
#     train_labels : array_like
#         Training labels, size (nImagesL&R,)
#     test_data : array_like
#         Testing data of size (nImagesToTest, nPixels*yPixels)
    
#     Keyword Arguments:
#     ------------------
#     m : int, optional
#         Scalar parameter (default: 1)
#     method : str, optional
#         Method type. Accepted values: 'PCA+LDA' (default: 'PCA+LDA')
#     model : dict or None, optional
#         Pre-trained model (default: None)
#     variance_to_keep : float, optional
#         Percentage of variance to keep for PCA (default: 95)
#     handle_nan : str, optional
#         How to handle NaN values: 'impute' (default), 'remove', or 'error'
        
#     Returns:
#     --------
#     pred : ndarray
#         Predictions for test data
#     model : dict
#         Trained model containing PCA and LDA components
#     """
    
#     # Parse keyword arguments
#     m = kwargs.get('m', 1)
#     method = kwargs.get('method', 'PCA+LDA')
#     model = kwargs.get('model', None)
#     variance_to_keep = kwargs.get('variance_to_keep', 95)
#     handle_nan = kwargs.get('handle_nan', 'impute')
    
#     # Convert to numpy arrays
#     train_data = np.array(train_data, dtype=np.float64)
#     train_labels = np.array(train_labels)
#     test_data = np.array(test_data, dtype=np.float64)
    
#     # Handle NaN values in labels first
#     nan_label_indices = np.isnan(train_labels)
#     if np.any(nan_label_indices):
#         print(f"Removing {np.sum(nan_label_indices)} samples with NaN labels")
#         train_labels = train_labels[~nan_label_indices]
#         train_data = train_data[~nan_label_indices]
    
#     # Handle NaN values in data
#     train_has_nan = np.isnan(train_data).any()
#     test_has_nan = np.isnan(test_data).any()
    
#     if train_has_nan or test_has_nan:
#         print(f"NaN values detected - Train: {np.isnan(train_data).sum()}, Test: {np.isnan(test_data).sum()}")
        
#         if handle_nan == 'error':
#             raise ValueError("NaN values found in data and handle_nan='error'")
        
#         elif handle_nan == 'remove':
#             # Remove samples with any NaN values
#             valid_train_mask = ~np.isnan(train_data).any(axis=1)
#             train_data = train_data[valid_train_mask]
#             train_labels = train_labels[valid_train_mask]
            
#             valid_test_mask = ~np.isnan(test_data).any(axis=1)
#             test_data = test_data[valid_test_mask]
            
#             print(f"Removed samples - Train: {np.sum(~valid_train_mask)}, Test: {np.sum(~valid_test_mask)}")
            
#         elif handle_nan == 'impute':
#             # Impute missing values
#             if model is None or 'imputer' not in model:
#                 # Fit imputer on training data
#                 imputer = SimpleImputer(strategy='mean')
#                 train_data = imputer.fit_transform(train_data)
#                 test_data = imputer.transform(test_data)
#                 store_imputer = True
#             else:
#                 # Use existing imputer
#                 imputer = model['imputer']
#                 train_data = imputer.transform(train_data)
#                 test_data = imputer.transform(test_data)
#                 store_imputer = False
#         else:
#             raise ValueError(f"Unknown handle_nan method: {handle_nan}")
    
#     # Check if we have enough samples after cleaning
#     if len(train_data) == 0:
#         raise ValueError("No valid training samples remaining after NaN handling")
    
#     # Check if we have at least 2 classes
#     unique_labels = np.unique(train_labels)
#     if len(unique_labels) < 2:
#         raise ValueError(f"Need at least 2 classes for classification, found {len(unique_labels)}")
    
#     # PCA+LDA method
#     if method == 'PCA+LDA':
#         if model is None or not isinstance(model, dict):
#             # Train new model
            
#             # PCA
#             pca = PCA()
#             pca_scores = pca.fit_transform(train_data)
#             pca_coefficients = pca.components_.T  # Transpose to match MATLAB convention
#             explained_variance_ratio = pca.explained_variance_ratio_
#             pca_centers = pca.mean_
            
#             # Determine number of components to keep
#             explained_variance_to_keep_as_fraction = variance_to_keep / 100
            
#             if variance_to_keep < 100:
#                 cumsum_explained = np.cumsum(explained_variance_ratio)
#                 num_components_to_keep = np.where(cumsum_explained >= explained_variance_to_keep_as_fraction)[0]
                
#                 if len(num_components_to_keep) == 0:
#                     # If we can't reach the desired variance, use all components
#                     print(f"Warning: Could not reach {variance_to_keep}% variance. Using all {len(explained_variance_ratio)} components.")
#                     num_components_to_keep = len(explained_variance_ratio)
#                 else:
#                     num_components_to_keep = num_components_to_keep[0] + 1
                
#                 pca_coefficients = pca_coefficients[:, :num_components_to_keep]
#                 train_predictors = pca_scores[:, :num_components_to_keep]
#             else:
#                 train_predictors = pca_scores
            
#             # LDA
#             mdl_linear = LinearDiscriminantAnalysis()
#             mdl_linear.fit(train_predictors, train_labels)
            
#             # Create model structure
#             model = {
#                 'pca_centers': pca_centers,
#                 'pca_coefficients': pca_coefficients,
#                 'mdl_linear': mdl_linear
#             }
            
#             # Add imputer to model if we used one
#             if handle_nan == 'impute' and store_imputer:
#                 model['imputer'] = imputer
                
#         else:
#             # Use existing model
#             pca_centers = model['pca_centers']
#             pca_coefficients = model['pca_coefficients']
#             mdl_linear = model['mdl_linear']
        
#         # Transform test data using PCA
#         test_data_pca = (test_data - pca_centers) @ pca_coefficients
        
#         # Make predictions
#         pred = mdl_linear.predict(test_data_pca)
    
#     else:
#         raise ValueError(f"Unknown method: {method}. Only 'PCA+LDA' is supported.")
    
#     return pred, model