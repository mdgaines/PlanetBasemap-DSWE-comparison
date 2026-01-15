import numpy as np
import geopandas as gpd
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

from sklearn.inspection import permutation_importance

from statistics import multimode


# Recursive Feature Elimination with Cross-Validation of a Random Forest
#########################################################################################################################
##### Modified from Marco Cerliani's answer on
##### https://stackoverflow.com/questions/62537457/right-way-to-use-rfecv-and-permutation-importance-sklearn
class RFExplainerRegressor(RandomForestClassifier):
    def fit(self, X,y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y
        )
        super().fit(X_train,y_train)
        
        self.perm_feature_importances_ = permutation_importance(
            self, X_val, y_val, 
            n_repeats=5, random_state=42,
        )['importances_mean']
        
        return super().fit(X,y)
#########################################################################################################################


# quick helper function to convert the class labels into indicies and 
# assign a dictionary relating the class indices to their names
def str_class_to_int(class_array):
    class_array[class_array == 'water'] = 1
    class_array[class_array == 'non-water'] = 0
    class_array[class_array == 'null'] = -1

    return(class_array.astype(int))


#########################################################################################################################
#############                                                                                               #############
#############                        Read in data for trainging and testing                                 #############
#############                              (i.e., accuracy assessment)                                      #############
#############                                                                                               #############
#########################################################################################################################

# read in training shp
training_data_shp = gpd.read_file('D:/Research/data/Shapefiles/train_classified_points.shp')

# upsample water points to match the number of non-water points

# diff between water and non-water points in water and non-water (N_diff)
N_diff = len(training_data_shp[training_data_shp['class']=='non-water']) - \
         len(training_data_shp[training_data_shp['class']=='water'])
# sample N_diff water points
N_diff_water_pnts = training_data_shp[training_data_shp['class']=='water'].sample(N_diff, random_state=42)
# add duplicated points to training df
training_data_gpd = gpd.pd.concat([training_data_shp, N_diff_water_pnts], ignore_index=True)

#### Get data into RandomForest format
bands = ['blue', 'green', 'red', 'nir', 
        'ndwi', 'ndvi', 
        'blue_3x3', 'green_3x3', 'red_3x3', 'nir_3x3', 
        'ndwi_3x3', 'ndvi_3x3',
        'elev', 'slope', 'hillshade', 
        'alpha']
X_t = np.array([])
for band in bands:
    point_vals = training_data_gpd[band].values
    if len(X_t) == 0:
        X_t = np.array(point_vals)
    else:
        # stack bands virtically
        X_t = np.vstack((X_t, point_vals))

X_t

X = X_t.T
Y = np.array(training_data_gpd['class'])
# X = np.delete(X, np.where(Y=='null')[0], axis=0)
# Y = np.delete(Y, np.where(Y=='null')[0])
# np.where(Y=='null')

# What are our classification labels?
labels = np.unique(Y)
print('The training data include {n} classes: {classes}\n'.format(n=labels.size, classes=labels))
print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=Y.shape))

Y_int = str_class_to_int(Y)


#########################################################################################################################
#############                                                                                               #############
#############           Feature Selection: Recursive Feature Elimination with Cross Valudation              #############
#############                              (RFECV and manually tested)                                      #############
#############                                                                                               #############
#########################################################################################################################


#### Testing Regressive Feature Elimination with Cross-Validation
model = RFExplainerRegressor(n_estimators=100, random_state=42)


# Using scikitlearn
selector = RFECV(model, step=1, min_features_to_select=1, 
                 importance_getter='perm_feature_importances_', 
                 scoring='roc_auc_ovo', cv=5, n_jobs=4)

selector.fit(X, Y_int)

features_selected = [list(i) for i in list(zip(bands, selector.get_support())) if i[1]]
print(features_selected)
features_selected2 = [[i[0][0], i[1]] for i in list(zip(features_selected, 
                                                        selector.estimator_.perm_feature_importances_))]
print(features_selected2)
df_rfecv_auc = pd.DataFrame(features_selected2, 
                            columns=['Band', 'PermutationImportance']).sort_values(
                                by=['PermutationImportance'], ascending=False)
df_rfecv_auc # going off of the bands selected here

'''
	Band	PermutationImportance
1	nir	        0.031210
8	elev    	0.019750
9	slope   	0.014341
6	ndwi_3x3    0.013028
2	ndwi	    0.008835
5	nir_3x3	    0.007810
7	ndvi_3x3	0.006402
10	hillshade	0.005762
3	ndvi	    0.004353
0	blue    	0.004193
4	blue_3x3	0.003297
'''

# Using scikitlearn
selector_f1 = RFECV(model, step=1, min_features_to_select=1, 
                 importance_getter='perm_feature_importances_', 
                 scoring='f1_macro', cv=5, n_jobs=4)

# Y_int = str_class_to_int(Y)
selector_f1.fit(X, Y_int)

features_selected_f1 = [list(i) for i in list(zip(bands, selector_f1.get_support())) if i[1]]
print(features_selected_f1)
features_selected_f12 = [[i[0][0], i[1]] for i in list(zip(features_selected_f1, 
                                                           selector_f1.estimator_.perm_feature_importances_))]
print(features_selected_f12)
df_rfecv_f1 = pd.DataFrame(features_selected_f12, 
                           columns=['Band', 'PermutationImportance']).sort_values(
                               by=['PermutationImportance'], ascending=False)
df_rfecv_f1

'''
	Band	PermutationImportance
1	nir	        0.307907
2	ndwi_3x3	0.113092
3	ndvi_3x3	0.031018
4	elev    	0.028585
5	slope	    0.020871
0	blue	    0.013796
'''

# Using scikitlearn
selector_accuracy = RFECV(model, step=1, min_features_to_select=1, 
                 importance_getter='perm_feature_importances_', 
                 scoring='accuracy', cv=5, n_jobs=4)

# Y_int = str_class_to_int(Y)
selector_accuracy.fit(X, Y_int)

features_selected_accuracy = [list(i) for i in list(zip(bands, selector_accuracy.get_support())) if i[1]]
print(features_selected_accuracy)
features_selected_accuracy2 = [[i[0][0], i[1]] for i in list(zip(features_selected_accuracy, 
                                                           selector_accuracy.estimator_.perm_feature_importances_))]
print(features_selected_accuracy2)
df_rfecv_accuracy = pd.DataFrame(features_selected_accuracy2, 
                           columns=['Band', 'PermutationImportance']).sort_values(
                               by=['PermutationImportance'], ascending=False)
df_rfecv_accuracy

'''
	Band	PermutationImportance
1	nir     	0.307907
2	ndwi_3x3	0.113092
3	ndvi_3x3	0.031018
4	elev	    0.028585
5	slope	    0.020871
0	blue    	0.013796
'''


# Manually

def find_least_important_feature(cv_results, bands):
    band_idx_lst = []
    for i in range(5):
        min_perm_feature_imp_val = cv_results['estimator'][i].perm_feature_importances_.min()
        band_idx_lst.append(np.where(cv_results['estimator'][i].perm_feature_importances_ == min_perm_feature_imp_val)[0][0])
        # band_lst.append(bands[band_idx_lst])
    print(f'least important band: {multimode(band_idx_lst)} {bands[multimode(band_idx_lst)[0]]}')
    return multimode(band_idx_lst)[0]

def print_cv_avg_scores(cv_results, count):
    test1_precision_macro_avg = cv_results['test_precision_macro'].mean()
    test1_recall_macro_avg = cv_results['test_recall_macro'].mean()
    test1_f1_macro_avg = cv_results['test_f1_macro'].mean()
    test1_accuracy_avg = cv_results['test_accuracy'].mean()
    test1_roc_auc_ovo_macro_avg = cv_results['test_roc_auc_ovo'].mean()
    print(f"\ntest results from v{count}\n\
         precision macro avg:\t{test1_precision_macro_avg}\n\
            recall macro avg:\t{test1_recall_macro_avg}\n\
                f1 macro avg:\t{test1_f1_macro_avg}\n\
                accuracy avg:\t{test1_accuracy_avg}\n\
             roc auc ovo avg:\t{test1_roc_auc_ovo_macro_avg}\
        ")
    return

def get_cv_results(model, X, Y_int):
    cv_results = cross_validate(model, X, Y_int, return_estimator=True,
                                scoring=['precision_macro',
                                        'recall_macro', 
                                        'f1_macro',
                                        'accuracy',
                                        'roc_auc_ovo'], cv=5)
    return cv_results

def manual_cv_feature_selection(model, X, Y_int, bands):
    # Step 1 A: check which band to remove
    cv_results = get_cv_results(model, X, Y_int)
    idx = find_least_important_feature(cv_results, bands)
    # Step 1 B: Get cv average scores
    print_cv_avg_scores(cv_results, 1)

    # Subset for set 2
    X_transformed_2 = np.delete(X.T, idx, axis=0).T
    # Step 2 A: check which band to remove
    cv_results2 = get_cv_results(model, X_transformed_2, Y_int)
    bands_lst_update2 = bands[:idx] + bands[idx+1:]
    idx2 = find_least_important_feature(cv_results2, bands_lst_update2)
    # Step 2 B: Get cv average scores
    print_cv_avg_scores(cv_results2, 2)

    # Subset for set 3
    X_transformed_3 = np.delete(X_transformed_2.T, idx2, axis=0).T
    # Step 3 A: check which band to remove
    cv_results3 = get_cv_results(model, X_transformed_3, Y_int)
    bands_lst_update3 = bands_lst_update2[:idx2] + bands_lst_update2[idx2+1:]
    idx3 = find_least_important_feature(cv_results3, bands_lst_update3)
    # Step 3 B: Get cv average scores
    print_cv_avg_scores(cv_results3, 3)

    # Subset for set 4
    X_transformed_4 = np.delete(X_transformed_3.T, idx3, axis=0).T
    # Step 4 A: check which band to remove
    cv_results4 = get_cv_results(model, X_transformed_4, Y_int)
    bands_lst_update4 = bands_lst_update3[:idx3] + bands_lst_update3[idx3+1:]
    idx4 = find_least_important_feature(cv_results4, bands_lst_update4)
    # Step 4 B: Get cv average scores
    print_cv_avg_scores(cv_results4, 4)

    # Subset for set 5
    X_transformed_5 = np.delete(X_transformed_4.T, idx4, axis=0).T
    # Step 5 A: check which band to remove
    cv_results5 = get_cv_results(model, X_transformed_5, Y_int)
    bands_lst_update5 = bands_lst_update4[:idx4] + bands_lst_update4[idx4+1:]
    idx5 = find_least_important_feature(cv_results5, bands_lst_update5)
    # Step 5 B: Get cv average scores
    print_cv_avg_scores(cv_results5, 5)

    # Subset for set 6
    X_transformed_6 = np.delete(X_transformed_5.T, idx5, axis=0).T
    # Step 6 A: check which band to remove
    cv_results6 = get_cv_results(model, X_transformed_6, Y_int)
    bands_lst_update6 = bands_lst_update5[:idx5] + bands_lst_update5[idx5+1:]
    idx6 = find_least_important_feature(cv_results6, bands_lst_update6)
    # Step 6 B: Get cv average scores
    print_cv_avg_scores(cv_results6, 6)

    # Subset for set 7
    X_transformed_7 = np.delete(X_transformed_6.T, idx6, axis=0).T
    # Step 7 A: check which band to remove
    cv_results7 = get_cv_results(model, X_transformed_7, Y_int)
    bands_lst_update7 = bands_lst_update6[:idx6] + bands_lst_update6[idx6+1:]
    idx7 = find_least_important_feature(cv_results7, bands_lst_update7)
    # Step 7 B: Get cv average scores
    print_cv_avg_scores(cv_results7, 7)

    # Subset for set 8
    X_transformed_8 = np.delete(X_transformed_7.T, idx7, axis=0).T
    # Step 8 A: check which band to remove
    cv_results8 = get_cv_results(model, X_transformed_8, Y_int)
    bands_lst_update8 = bands_lst_update7[:idx7] + bands_lst_update7[idx7+1:]
    idx8 = find_least_important_feature(cv_results8, bands_lst_update8)
    # Step 8 B: Get cv average scores
    print_cv_avg_scores(cv_results8, 8)

    # Subset for set 9
    X_transformed_9 = np.delete(X_transformed_8.T, idx8, axis=0).T
    # Step 9 A: check which band to remove
    cv_results9 = get_cv_results(model, X_transformed_9, Y_int)
    bands_lst_update9 = bands_lst_update8[:idx8] + bands_lst_update8[idx8+1:]
    idx9 = find_least_important_feature(cv_results9, bands_lst_update9)
    # Step 9 B: Get cv average scores
    print_cv_avg_scores(cv_results9, 9)

    # Subset for set 10
    X_transformed_10 = np.delete(X_transformed_9.T, idx9, axis=0).T
    # Step 10 A: check which band to remove
    cv_results10 = get_cv_results(model, X_transformed_10, Y_int)
    bands_lst_update10 = bands_lst_update9[:idx9] + bands_lst_update9[idx9+1:]
    idx10 = find_least_important_feature(cv_results10, bands_lst_update10)
    # Step 10 B: Get cv average scores
    print_cv_avg_scores(cv_results10, 10)

    return [cv_results, 
            cv_results2, bands_lst_update2,
            cv_results3, bands_lst_update3,
            cv_results4, bands_lst_update4,
            cv_results5, bands_lst_update5,
            cv_results6, bands_lst_update6,
            cv_results7, bands_lst_update7,
            cv_results8, bands_lst_update8,
            cv_results9, bands_lst_update9,
            cv_results10, bands_lst_update10]

##### Print everything again #####

[cv_results, 
    cv_results2, bands_lst_update2,
    cv_results3, bands_lst_update3,
    cv_results4, bands_lst_update4,
    cv_results5, bands_lst_update5,
    cv_results6, bands_lst_update6,
    cv_results7, bands_lst_update7,
    cv_results8, bands_lst_update8,
    cv_results9, bands_lst_update9,
    cv_results10, bands_lst_update10] = manual_cv_feature_selection(model, X, Y_int, bands)

# v1
find_least_important_feature(cv_results, bands)
print_cv_avg_scores(cv_results, 1)
# v2
find_least_important_feature(cv_results2, bands_lst_update2)
print_cv_avg_scores(cv_results2, 2)
# v3
find_least_important_feature(cv_results3, bands_lst_update3)
print_cv_avg_scores(cv_results3, 3)
# v4
find_least_important_feature(cv_results4, bands_lst_update4)
print_cv_avg_scores(cv_results4, 4)
# v5
find_least_important_feature(cv_results5, bands_lst_update5)
print_cv_avg_scores(cv_results5, 5)
# v6
find_least_important_feature(cv_results6, bands_lst_update6)
print_cv_avg_scores(cv_results6, 6)
# v7
find_least_important_feature(cv_results7, bands_lst_update7)
print_cv_avg_scores(cv_results7, 7)
# v8
find_least_important_feature(cv_results8, bands_lst_update8)
print_cv_avg_scores(cv_results8, 8)
# v9
find_least_important_feature(cv_results9, bands_lst_update9)
print_cv_avg_scores(cv_results9, 9)
# v10
find_least_important_feature(cv_results10, bands_lst_update10)
print_cv_avg_scores(cv_results10, 10)


#########################################################################################################################
#############                                                                                               #############
#############                          Hyperparameter Tuning with Cross Validation                          #############
#############                                          (GridSearchCV)                                       #############
#############                                                                                               #############
#########################################################################################################################

# Subset bands to those selected in Feature Selection
# index_bands = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
index_bands = [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
bands_subset = [bands[i] for i in index_bands]

# index_bands_remove = [1, 7, 15]
index_bands_remove = [1, 2, 7, 8, 15]
X_11_bands = np.delete(X.T, index_bands_remove, axis=0).T

parameters = {
    'n_estimators' : [50, 100, 200, 300, 500, 700],
    'max_features' : [2, 3, 5]
}

rf_gridSearch = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=parameters,
                             scoring=['precision_macro', 'recall_macro', 'f1_macro',
                                      'accuracy', 'roc_auc_ovo'], 
                             cv=5, n_jobs=4, refit='roc_auc_ovo')

rf_gridSearch.fit(X_11_bands, Y_int)

mean_scores_dict = dict([[i, rf_gridSearch.cv_results_[i]] \
    for i in rf_gridSearch.cv_results_ if 'mean' in i and 'test' in i])

# scores are: 
#   'mean_test_precision_macro', 
#   'mean_test_recall_macro', 
#   'mean_test_f1_macro', 
#   'mean_test_accuracy', 
#   'mean_test_roc_auc_ovo'

for key in mean_scores_dict.keys():
    key_idx = np.where(mean_scores_dict[key] == mean_scores_dict[key].max())[0][0]
    print(f"{key} etimator: {key_idx} params: {rf_gridSearch.cv_results_['params'][key_idx]}")
    for key2 in mean_scores_dict.keys():
        print(f"\t{key2} :\t\t{mean_scores_dict[key2][key_idx]}")

### Decision: use the params that were most frequently performing the best when looking at all 5 scoring metrics
# using features selected from ROC AUC OVO
### params: max_features = 3, n_estimators = 100

