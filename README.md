                                PREDICTING AGE OF AN ABALONE FROM ITS PHYSICAL MEASUREMENTS USING A DEEP NEURAL NETWORK

Abalones are marine snails. There are between 30 to 130 types of abalones on earth. Abalones vary in sizes between 20 mm to 200mm. The shell of a greater part of their species 
is raised, oval and may be angled or straightened. 

The age of abalone is predicted by first cutting open the shell and then tallying the number of rings (add 1.5 to number of rings for deriving age in years) with a magnifying
lens, which is a tedious procedure. Accordingly, some simple to-utilize features are used to anticipate its age. 

A Data set having 4177 samples was downloaded from UCI online library. In the dataset, continuous value measurements have been scaled by a factor of 200.

 **Dataset Features** 
 
1. Sex - Male, Female and Infant 
2. 	Length ( mm) - Longest shell dimension
3. 	Width (mm) - Perpendicular to length 
4. 	Stature (mm)- With meat in shell 
5. 	Entire weight (grams) - Whole abalone 
6. 	Shucked weight (grams) - Weight of meat 
7. 	Viscera weight (grams) - Gut weight (subsequent to dying) 
8. 	Shell weight (grams) - After being dried 
9.	Rings (number) ±1.5 gives age in years

**Machine Learning Libraries Used**

1. Keras (higher level API for deep learning with Tensorflow in backend)
2. Tensorflow (deep learning library in backend)
3. AutoKeras(Auto Machine Learning library)
4. Pandas(for data processing)
5. SciKit Learn(for data preparation) 
6. Matplotlib(for plotting graphs)

Pycharm is the IDE used.  

 
**Implementation**

1. Importing libraries (pandas, numpy, matplotlib, autokeras) 
2. Reading the data in a data frame. 
3.Gathering statistical information from the data   (unique values, numerical and categorical data)
4. Cleaning the dataset (removing outliers)
5. Splitting the data into training, validation and test set
6. Dividing the data set into numerical and categorical features
7. Feature selection, feature engineering, scaling and PCA (on numerical features to reduce noise)
8. One Hot Encoding (categorical features)
9. Concatenating numerical and categorical features
10. Scaling the targets
11. Training the model
12. Prepare the test data set
13. Evaluate the model on the test data


 **Result**

The ANN model (containing 2 ANN layers) generated a mean absolute error (MAE) of **1.59 rings** (i.e age of the abalone = +1.5 𝑦𝑟𝑠  in the test data set.)
