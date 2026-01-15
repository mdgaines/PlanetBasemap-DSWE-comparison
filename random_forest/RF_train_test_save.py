import joblib
import numpy as np
import geopandas as gpd
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# quick helper function to convert the class labels into indicies and 
# assign a dictionary relating the class indices to their names
def str_class_to_int(class_array):
    '''
    Converts dependent variable (Y) from string classes to integers.
    
    :param class_array: Array of class information
    '''
    class_array[class_array == 'water'] = 1
    class_array[class_array == 'non-water'] = 0
    class_array[class_array == 'null'] = -1

    return(class_array.astype(int))


def get_rf_features(data_path:str, train:bool):
    '''
    Returns the features to be used in training or testing the random forest.
        IF TRAINING: Upsamples the number of water values to match the number of non-water values.
        Removes bands that we are not using (based on feature selection process).
        Converts dependent variable (Y) from string classes to integers.
    
    :param data_path: Path to the testing or training data.
    :type data_path: str
    :param train: Boolean to flag if data is trainnig (TRUE) or testing (FALSE)
    :type train: bool
    '''

    data_gpd = gpd.read_file(data_path)

    if train:
        # upsample water points to match the number of non-water points
        # diff between water and non-water points in water and non-water (N_diff)
        N_diff = len(data_gpd[data_gpd['class']=='non-water']) - \
                len(data_gpd[data_gpd['class']=='water'])
        # sample N_diff water points
        N_diff_water_pnts = data_gpd[data_gpd['class']=='water'].sample(N_diff, random_state=42)
        # add duplicated points to training df
        data_gpd = gpd.pd.concat([data_gpd, N_diff_water_pnts], ignore_index=True)
        data_set = 'train'
    else:
        data_set = 'test'


    #### Get data into RandomForest format
    bands = ['blue', 'green', 'red', 'nir', 
            'ndwi', 'ndvi', 
            'blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3', 
            'ndwi_3x3', 'ndvi_3x3',
            'elev', 'slope', 'hillshade', 
            'alpha']
    
    X_t = np.array([])
    for band in bands:
        point_vals = data_gpd[band].values
        if len(X_t) == 0:
            X_t = np.array(point_vals)
        else:
            # stack bands virtically
            X_t = np.vstack((X_t, point_vals))

    X_full = X_t.T
    # subset to the 11 bands we are using
    index_bands_remove = [1, 2, 7, 8, 15]

    X = np.delete(X_full.T, index_bands_remove, axis=0).T
    # convert from string to int for RF Classifier
    Y_str = np.array(data_gpd['class'])
    labels = np.unique(Y_str)

    label_dict = {'water': 1, 'non-water': 0, 'null': -1}

    Y = str_class_to_int(Y_str)

    # What are our classification labels?
    if labels.size == 3:
        print(f'\nThe {data_set} data include {labels.size} classes: {labels} aka {\
                [label_dict[labels[0]], label_dict[labels[1]], label_dict[labels[2]]]}\n')
    elif labels.size == 2:
        print(f'\nThe {data_set} data include {labels.size} classes: {labels} aka {\
                [label_dict[labels[0]], label_dict[labels[1]]]}\n')
    print(f'Our X matrix is sized: {X.shape}')
    print(f'Our Y array is sized: {Y.shape}')

    return X, Y


def main():
    #########################################################################################################################
    #############                                                                                               #############
    #############                        Read in data for trainging and testing                                 #############
    #############                              (i.e., accuracy assessment)                                      #############
    #############                                                                                               #############
    #########################################################################################################################

    # Training dataset
    X_train, Y_train = get_rf_features('D:/Research/data/Shapefiles/train_classified_points.shp', train=True)
    # Testing dataset
    X_test, Y_test = get_rf_features('D:/Research/data/Shapefiles/test_classified_points.shp', train=False)

    # Print test Y variable split
    label_dict = {1:'water', 0:'non-water', -1:'null'}
    val_array, count_array = np.unique(Y_test, return_counts=True)
    print(f"test value counts")
    for i in range(3):
        print(f"{label_dict[val_array[i]]} : {count_array[i]}")

    #########################################################################################################################
    #############                                                                                               #############
    #############                            Run, test, and save final Random Forest                            #############
    #############                                                                                               #############
    #########################################################################################################################

    # Subset bands to those selected in Feature Selection
    bands_full = ['blue', 'green', 'red', 'nir', 
            'ndwi', 'ndvi', 
            'blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3', 
            'ndwi_3x3', 'ndvi_3x3', 
            'elev', 'slope', 'hillshade',
            'alpha']
    index_bands = [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14] # testing from RFECV ROC AUC

    bands = [bands_full[i] for i in index_bands]

    print('\nTraining Features Shape:', X_train.shape)
    print('Training Labels Shape:', Y_train.shape)
    print('\nTesting Features Shape:', X_test.shape)
    print('Testing Labels Shape:', Y_test.shape)

    # Instantiate model with 100 decision trees
    # rf = RandomForestClassifier(n_estimators = 200, max_features=2, random_state = 42)
    rf = RandomForestClassifier(n_estimators = 100, max_features=3, random_state = 42)

    # Fit the model on training data
    rf.fit(X_train, Y_train)

    # Predicting the Test set results
    class_pred = rf.predict(X_test)


    print(confusion_matrix(Y_test, class_pred))

    ConfusionMatrixDisplay.from_predictions(Y_test, class_pred)
    plt.show()

    # Accuracy Score
    print(accuracy_score(Y_test, class_pred))

    # Classificaton Report 
    print(classification_report(Y_test, class_pred, digits=4))


    ## Look at training to check for overfitting
    # Predicting the train set results
    class_pred_train = rf.predict(X_train)


    print(confusion_matrix(Y_train, class_pred_train))

    ConfusionMatrixDisplay.from_predictions(Y_train, class_pred_train)
    plt.show()

    # Accuracy Score
    print(accuracy_score(Y_train, class_pred_train)) # 0.9999260245598461 for all but green and green_3x3

    # Classificaton Report 
    print(classification_report(Y_train, class_pred_train, digits=4))


    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # print(importances)
    df = pd.DataFrame({'Band':bands,'RF_Importance':importances}).sort_values(by=['RF_Importance'],ascending=False)
    df

    # Get the permutation feature importances
    perm_feature_importances = permutation_importance(rf, X_test, Y_test, 
                                                    n_repeats=10, random_state=42)['importances_mean']
    df2 = pd.DataFrame({'Band':bands,'RF_Perm_Importance':perm_feature_importances})\
                        .sort_values(by=['RF_Perm_Importance'],ascending=False)
    df2

    # Get the permutation feature importances
    perm_feature_importances_train = permutation_importance(rf, X_train, Y_train, 
                                                            n_repeats=10, random_state=42)['importances_mean']
    df3 = pd.DataFrame({'Band':bands,'RF_Perm_Importance':perm_feature_importances_train})\
                        .sort_values(by=['RF_Perm_Importance'],ascending=False)
    df3

    # Save Random Forest
    joblib.dump(rf, "D:/Research/data/RF_SRTM_AUC_compressed.joblib", compress=3)

    return

if __name__ == '__main__':
    main()