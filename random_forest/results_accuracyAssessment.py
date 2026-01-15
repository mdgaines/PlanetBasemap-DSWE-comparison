# !/usr/bin/env python
import re, os, shutil, sys
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from random_forest.RF_train_test_save import get_rf_features
from random_forest.compile_classified_points import compile_shp
import matplotlib.pyplot as plt
import joblib
import geopandas as gpd
import pandas as pd

INIT_PATH = 'D:/Research/'

######################################################################################################
#### Modified from https://github.com/suryakant54321/accuracyAssess/blob/master/accessAccuracy.py ####
######################################################################################################

errStatus0 = "enter complete file path"
#
# if len(sys.argv)==2:
# 	filePath = sys.argv[1] #os.chdir("../accuracyAccess/errorMat.txt")
# 	print(filePath)
# else:
# 	print(errStatus0)
# Check errors in input file
errStatus = "Problem with input text file \n recheck the input again \n Suggested format comma seperated values : \n First line number of pixels for each class \n Next lines error matrix. "

def checkErrors(areaArray, errorMat):
	status = "TRUE"	
	if(len(areaArray)==len(errorMat)):
		status = "TRUE"
	else:
		status = errStatus 
	return status

# main
def analyzeAccuracy(filePath, fName):
	# fName = "output.txt"
	curPath = os.getcwd()
	print(f"\n\nwriting output to \n {curPath}/{fName}")	
	fName = open(fName, 'w')
	areaArray = []
	errorMat = []
	count = 0
	for i in open(filePath):
		row = re.split(',',i)
		row[-1] = row[-1].rstrip('\n')
		if (count == 0):
			areaArray = row
		else:
			errorMat.append(row)
		count = count + 1
	try:	
		areaArray = np.asarray(areaArray, dtype=np.float32)
		fName.write("User Inputs \n 1) Area Distribution \n")
		np.savetxt(fName, areaArray, delimiter=',', fmt='%1.4f')
		#fName.write(areaArray)
		print(areaArray)
		totalArea = areaArray.sum()
		errorMat = np.asarray(errorMat, dtype=np.float32)
		fName.write("\n 2) Confustion matrix \n")		
		np.savetxt(fName, errorMat, delimiter=',', fmt='%1.4f')
		print(errorMat)
	except:
		print(errStatus)
	# Check the errors in user input
	check = checkErrors(areaArray, errorMat)
	if (check=="TRUE"):
		# 
		rowSum = []
		colSum = []
		for i in range (0,(len(errorMat))):
			rowSum.append(errorMat[i,:].sum())
			colSum.append(errorMat[:,i].sum())
		#
		rowSum = np.asarray(rowSum, dtype=np.float32)
		colSum = np.asarray(colSum, dtype=np.float32)
		# print(f"Total Area [in px] = {totalArea}") 
		# print(f"Sum of each Row = {rowSum}")
		# print(f"Sum of each Column = {colSum}")
		# -- fun begins
		wi = areaArray/totalArea
		wiPerCent = wi*100
		# print(f"Wi = {wi}")
		print(f"Wi (in percent)= {wiPerCent}")
		#-------------------------------------------------
		# To do 
		# estimate sample size for each class
		#-------------------------------------------------
		# Error Matrix estiamted area proportions
		erMatEAP = []
		for i in range (0,(len(errorMat))):
			erMatEAP.append((errorMat[i,]*wi[i])/rowSum[i])
		erMatEAP = np.asarray(erMatEAP, dtype=np.float32)
		# print("Error Matrix estiamted area proportions")
		# print(erMatEAP)
		# 
		erroRowSum = []
		errorColSum = []
		for i in range (0,(len(erMatEAP))):
			errorColSum.append(erMatEAP[i,:].sum())
			erroRowSum.append(erMatEAP[:,i].sum())
		# 
		print(f"error Column \n {errorColSum}")
		print(f"error Row Area ^ \n {erroRowSum}")

		areaPix = (erroRowSum * np.asarray(totalArea, dtype=np.float32))
		# print(f"Area pix {areaPix}")
		haConv = (100*100)
		pixConv = (4.77*4.77)
		areaHa = ((areaPix * np.asarray(pixConv, dtype=np.float32))/np.asarray(haConv, dtype=np.float32))
		# print(f"Area in ha {areaHa}")
		# 
		pij = erMatEAP
		ni = rowSum
		# pij	
		# ni	
		# wi	
		print(f"pij \n {pij}")
		fName.write("\n Analysis Output \n 1) Matrix p(ij) \n")
		np.savetxt(fName, pij, delimiter=',', fmt='%1.4f')		
		# print(f"ni \n {ni}")
		fName.write("\n 2) Number of pixels MAPPED to class i; n(i) \n")
		np.savetxt(fName, ni, delimiter=',', fmt='%1.4f')		
		# print(f"wi \n {wi}")
		fName.write("\n 3) Weight values Wi \n")
		np.savetxt(fName, wi, delimiter=',', fmt='%1.4f')
		SD = []
		for i in range (0,(len(pij))):
			ss = ((pij[:,i]*wi-(pij[:,i]*pij[:,i]))/(ni))
			ss = np.sqrt(ss.sum())
			SD.append(ss)
		#
		SD = np.asarray(SD, dtype=np.float32)
		fName.write("\n 4) Standard Error (SE) \n")
		np.savetxt(fName, SD, delimiter=',', fmt='%1.4f')
		# print(f"Standard Error = {SD}")
		SdPix = SD* np.asarray(totalArea, dtype=np.float32)
		fName.write("\n 5) Standard Error in [ px ] \n")
		np.savetxt(fName, SdPix, delimiter=',', fmt='%1.4f')
		# print(f"Standard Error in [ px ] = {SdPix}")
		SdInHa = ((SdPix * np.asarray(pixConv, dtype=np.float32))/np.asarray(haConv, dtype=np.float32))
		# print(f"Standard Error in [ ha ] = {SdInHa}")
		fName.write("\n 6) Standard Error in [ ha ] \n")
		np.savetxt(fName, SdInHa, delimiter=',', fmt='%1.4f')
		# for 95 % confidence interval 1.98
		CiVal = 1.98
		ConfInt = SdInHa * np.asarray(CiVal, dtype=np.float32)
		# print(f"95 CI in [ ha ] = {ConfInt}")
		fName.write("\n 7) Constant for 95 percent Confidence Interval (CI) = ")
		valC = ("%s")%(CiVal)
		fName.write(valC)
		fName.write("\n \n 8) CI in [ ha ] \n")
		np.savetxt(fName, ConfInt, delimiter=',', fmt='%1.4f')
		#-------------------------------------------------
		# Margin of error
		MoE = (ConfInt / areaHa)
		# print(f"Margin of Error = {MoE}")
		fName.write("\n 9) Margin of Error (MoE) \n  \n")
		np.savetxt(fName, MoE, delimiter=',', fmt='%1.4f')
		MoEPerCent = MoE* np.asarray(100, dtype=np.float32)
		# print(f"Margin of Error in Percent = {MoEPerCent}")
		fName.write("\n 10) MoE in [ percent ] \n  \n")
		np.savetxt(fName, MoEPerCent, delimiter=',', fmt='%1.4f')
		#-------------------------------------------------
		# Overall Accuracy
		OverallA = np.diag(pij)
		OverallA = OverallA.sum()
		print(f"Overal accuracy = {OverallA}")
		fName.write("\n 11) Overall accuracy \n  \n")
		ov = ("%s")%(OverallA)		
		fName.write(ov)
		#-------------------------------------------------
		# User's accuracy
		UAccuracy = (np.diag(pij)/wi)
		print(f"User's accuracy = {UAccuracy}")
		fName.write("\n 12) User's accuracy \n  \n")
		np.savetxt(fName, UAccuracy, delimiter=',', fmt='%1.4f')
		#-------------------------------------------------
		# Producers's accuracy
		ProdAccuracy = (np.diag(pij)/erroRowSum)
		print(f"Producers Accuracy = \n {ProdAccuracy}")
		fName.write("\n 13) Producers Accuracy \n  \n")
		np.savetxt(fName, ProdAccuracy, delimiter=',', fmt='%1.4f')
		#-------------------------------------------------
		fName.close()
		#

		n_00 = np.round(pij[0][0], decimals=4)
		n_01 = np.round(pij[0][1], decimals=2)
		n_02 = np.round(errorColSum[0], decimals=2)
		n_03 = np.round(UAccuracy[0], decimals=2)
		print(n_00)

		print((f"\nFull Confusion Matrix\n"
		 	   f"         			      Reference       \n"
			   f"			Non-Water\t\tWater\t\t\tTotal\t\t\tUA\n"
			   f"\tNon-water	{pij[0][0]}\t{pij[0][1]}\t{errorColSum[0]}\t{UAccuracy[0]}\n"
			   f"\t    Water	{pij[1][0]}\t{pij[1][1]}\t{errorColSum[1]}\t{UAccuracy[1]}\n"
			   f"\t    Total	{erroRowSum[0]}\t{erroRowSum[1]}\t100\n"
			   f"\t       PA 	{ProdAccuracy[0]}\t{ProdAccuracy[1]}\n\n"			   
			   ))

		return UAccuracy, ProdAccuracy
	else:
		print(check)

######################################################################################################
######################################################################################################

def get_cluster_standardError(shp_path, Ui=[], Pj=[]):
	'''
		Based on formulas in Stehman et al., 2021 and Stehman et al., 1997
	'''

	# for full dataset
	if os.path.basename(shp_path) == 'test_classified_points.shp':
		AA_df = pd.read_table('./accuracy_assessments/full_AccuracyAssessment.txt')
		szn = 'full'
	else:
		szn = os.path.basename(shp_path).split('_')[0]
		AA_df = pd.read_table(f'./accuracy_assessments/{szn}_AccuracyAssessment.txt')
	
	se_outpath = f"./accuracy_assessments/{szn}_ClusterStandardError.txt"
	if os.path.exists(se_outpath):
		print(f"{se_outpath} exists")
		return

	print(f"\n\nGetting clustered Standard Error for {szn}")

	X_test, Y_test = get_rf_features(shp_path, train=False)
	rf = joblib.load('D:/Research/data/RF_SRTM_AUC_compressed.joblib')
	class_pred = rf.predict(X_test)

	test_full = gpd.read_file(shp_path)
	test_full['mapped_class'] = class_pred
	unique_points = list((set(test_full['geometry'])))
	m = len(unique_points)

	# user's accuracy for i0 and i1
	# non-water and water
	# Ui = [0.9777778, 0.9324324]
	if len(Ui) < 2:
		Ui = [float(AA_df.iloc[38].values[0]), float(AA_df.iloc[39].values[0])]
	# producer's accuracy for j0 and j1
	# non-water and water
	# Pj = [0.9942734, 0.7776388]
	if len(Pj) < 2:
		Pj = [float(AA_df.iloc[41].values[0]), float(AA_df.iloc[42].values[0])]

	class_dict = {0: 'non-water',
			   	  1: 'water'}

	SUM_ni_dot = np.array([0, 0])
	SUM_nj_dot = np.array([0, 0])
	ni_dot_u = np.array([[0, 0]] * m)
	nj_dot_u = np.array([[0, 0]] * m)
	nii_u = np.array([[0, 0]] * m)
	njj_u = np.array([[0, 0]] * m)
	p_u = np.array([0] * m)
	SUM_pu = 0
	for u_idx in range(len(unique_points)):
		# pixel u
		u = unique_points[u_idx]
		u_info = test_full[test_full['geometry'] == u]

		ni0u = len(u_info[u_info['mapped_class'] == 0])
		SUM_ni_dot[0] += ni0u
		ni1u = len(u_info[u_info['mapped_class'] == 1])
		SUM_ni_dot[1] += ni1u
		
		ni_dot_u[u_idx][0] = ni0u
		ni_dot_u[u_idx][1] = ni1u

		nj0u = len(u_info[u_info['class'] == class_dict[0]])
		SUM_nj_dot[0] += nj0u
		nj1u = len(u_info[u_info['class'] == class_dict[1]])
		SUM_nj_dot[1] += nj1u

		nj_dot_u[u_idx][0] = nj0u
		nj_dot_u[u_idx][1] = nj1u

		nii_u[u_idx][0] = len(u_info[(u_info['mapped_class'] == 0) & 
							   (u_info['class'] == class_dict[0])])
		nii_u[u_idx][1] = len(u_info[(u_info['mapped_class'] == 1) & 
							   (u_info['class'] == class_dict[1])])
		
		njj_u[u_idx][0] = len(u_info[(u_info['mapped_class'] == 0) & 
							   (u_info['class'] == class_dict[0])])
		njj_u[u_idx][1] = len(u_info[(u_info['mapped_class'] == 1) & 
							   (u_info['class'] == class_dict[1])])
		
		p_u[u_idx] = (len(u_info[(u_info['mapped_class'] == 0) & 
							   (u_info['class'] == class_dict[0])]) + \
					  len(u_info[(u_info['mapped_class'] == 1) & 
							   (u_info['class'] == class_dict[1])]) ) / len(u_info)
		
		SUM_pu += p_u[u_idx]

	ni_dot = SUM_ni_dot / m
	nj_dot = SUM_nj_dot / m
	p_bar = SUM_pu / m

	se_Ui = (1/ni_dot) *  np.sqrt(np.sum((nii_u - Ui * ni_dot_u) ** 2, axis=0) / (m * (m-1)))
	se_Pj = (1/nj_dot) *  np.sqrt(np.sum((njj_u - Pj * nj_dot_u) ** 2, axis=0) / (m * (m-1)))
	se_O = np.sqrt((np.sum((p_u - p_bar) ** 2 )) / (m * (m-1)))

	print(f"Standard Error of the User's Accuracy:     {se_Ui * 100} %")
	print(f"Standard Error of the Producer's Accuracy: {se_Pj * 100} %")
	print(f"Standard Error of the Overall Accuracy:    {se_O * 100} %")

	fName = open(se_outpath, 'w')
	fName.write(f"Standard Error of the User's Accuracy:     {se_Ui * 100} %\n")
	fName.write(f"Standard Error of the Producer's Accuracy: {se_Pj * 100} %\n")
	fName.write(f"Standard Error of the Overall Accuracy:    {se_O * 100} %\n")
	fName.close()

	return

def get_test_data():
	# Testing dataset
	# full
	X_test, Y_test = get_rf_features('D:/Research/data/Shapefiles/test_classified_points.shp', train=False)

	# spring test
	compile_shp(lst=[f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20170327_20170403_mosaic.shp',
					f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20190318_20190325_mosaic.shp'],
					outpath='spring_test_class_points.shp')
	X_test_spring, Y_test_spring = get_rf_features('D:/Research/data/Shapefiles/spring_test_class_points.shp', train=False)

	# summer test
	compile_shp(lst=[f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20170821_20170828_mosaic.shp',
					f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20190722_20190729_mosaic.shp'],
					outpath='summer_test_class_points.shp')
	X_test_summer, Y_test_summer = get_rf_features('D:/Research/data/Shapefiles/summer_test_class_points.shp', train=False)

	# fall test
	compile_shp(lst=[f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20170925_20171002_mosaic.shp',
					f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20190923_20190930_mosaic.shp'],
					outpath='fall_test_class_points.shp')
	X_test_fall, Y_test_fall = get_rf_features('D:/Research/data/Shapefiles/fall_test_class_points.shp', train=False)

	# winter test
	compile_shp(lst=[f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20180108_20180115_mosaic.shp'],
					#  f'{INIT_PATH}data/PlanetBasemaps/mosaic_shapefiles\\20200217_20200224_mosaic.shp'],
					outpath='winter_test_class_points.shp')
	X_test_winter, Y_test_winter = get_rf_features('D:/Research/data/Shapefiles/winter_test_class_points.shp', train=False)

	return X_test, Y_test, X_test_spring, Y_test_spring, \
			X_test_summer, Y_test_summer, \
			X_test_fall, Y_test_fall, \
			X_test_winter, Y_test_winter

def get_raw_accuracy_info(rf, X_test, Y_test):

	# Predicting the Test set results
	class_pred = rf.predict(X_test)

	print(confusion_matrix(Y_test, class_pred))

	ConfusionMatrixDisplay.from_predictions(Y_test, class_pred)
	plt.show()

	# Accuracy Score
	print(accuracy_score(Y_test, class_pred))

	# Classificaton Report 
	print(classification_report(Y_test, class_pred, digits=4))


def main():
	X_test, Y_test, X_test_spring, Y_test_spring, \
			X_test_summer, Y_test_summer, \
			X_test_fall, Y_test_fall, \
			X_test_winter, Y_test_winter = get_test_data()
	
	rf = joblib.load('D:/Research/data/RF_SRTM_AUC_compressed.joblib')

	# full
	get_raw_accuracy_info(rf, X_test, Y_test)
	# spring
	get_raw_accuracy_info(rf, X_test_spring, Y_test_spring)
	# summer
	get_raw_accuracy_info(rf, X_test_summer, Y_test_summer)
	# fall
	get_raw_accuracy_info(rf, X_test_fall, Y_test_fall)
	# winter
	get_raw_accuracy_info(rf, X_test_winter, Y_test_winter)

	# Implementation

	# full
	Ui, Pj = analyzeAccuracy("./confusion_matrix/full_confusion_matrix.txt", "./accuracy_assessments/full_AccuracyAssessment.txt")
	get_cluster_standardError(shp_path='D:/Research/data/Shapefiles/test_classified_points.shp', Ui=Ui, Pj=Pj)
	
	# seasonal
	Ui, Pj = analyzeAccuracy("./confusion_matrix/spring_conf_matrix.txt", "./accuracy_assessments/spring_AccuracyAssessment.txt")
	get_cluster_standardError(shp_path='D:/Research/data/Shapefiles/spring_test_class_points.shp', Ui=Ui, Pj=Pj)
	
	Ui, Pj = analyzeAccuracy("./confusion_matrix/summer_conf_matrix.txt", "./accuracy_assessments/summer_AccuracyAssessment.txt")
	get_cluster_standardError(shp_path='D:/Research/data/Shapefiles/summer_test_class_points.shp', Ui=Ui, Pj=Pj)
	
	Ui, Pj = analyzeAccuracy("./confusion_matrix/fall_conf_matrix.txt", "./accuracy_assessments/fall_AccuracyAssessment.txt")
	get_cluster_standardError(shp_path='D:/Research/data/Shapefiles/fall_test_class_points.shp', Ui=Ui, Pj=Pj)
	
	Ui, Pj = analyzeAccuracy("./confusion_matrix/winter_conf_matrix.txt", "./accuracy_assessments/winter_AccuracyAssessment.txt")
	get_cluster_standardError(shp_path='D:/Research/data/Shapefiles/winter_test_class_points.shp', Ui=Ui, Pj=Pj)

	return

if __name__ == '__main__':
    main()
