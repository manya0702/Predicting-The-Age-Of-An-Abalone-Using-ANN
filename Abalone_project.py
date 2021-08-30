import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import autokeras as ak

df = pd.read_csv('C:/Users/manya/PycharmProjects/pythonProject/abalone/abalone.txt')
df.columns = ['sex','length','diameter','height','whole_wight','shucked_weight','viscera_weight','shell_weight','rings']

df.head()
df.info()
df.describe()
df.nunique()
df.duplicated().any() # detect duplicate rows

y = df.iloc[:,-1]
X = df.iloc[:,:-1]
print(X.shape, y.shape)

plt.boxplot(X.iloc[:,1:])
plt.show()

# Isolation Forest :Auto remove outliers in dataset Num features
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05) #proportion of outliers in the dataset range 0 to 0.5
yhat = iso.fit_predict(X.iloc[:,1:])
# select all rows that are not outliers
mask = yhat != -1
X_masked, y_masked = X[mask], y[mask]
X, y = X_masked, y_masked
# summarize the shape of the updated dataset
print(X.shape, y.shape)
# Verify outliers auto removed in dataset Num features
plt.boxplot(X_masked.iloc[:,1:])
plt.show()

# split into train, eval and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# spit into num and cat features
X_train_num = X_train.iloc[:,1:]
X_val_num = X_val.iloc[:,1:]
X_test_num = X_test.iloc[:,1:]
X_train_cat = X_train.iloc[:,0]
X_val_cat = X_val.iloc[:,0]
X_test_cat = X_test.iloc[:,0]

# Num Feature Selection : Pearson's Correation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
fs = SelectKBest(score_func=f_regression, k = 'all')
fs.fit(X_train_num, y_train)
# plot
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

# Histgram to see distribution
fig = X_train_num.hist(xlabelsize=2,ylabelsize=2 )
plt.show()

# Feature Engineering : Create new Polynomial features 
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree=2) # Degree of polynomial features created
pol.fit(X_train_num)
X_train_num_engg = pol.transform(X_train_num)
X_val_num_engg = pol.transform(X_val_num)
print(X_train_num_engg.shape)
print(X_train_num_engg[0:5])

# Data Scaling
from sklearn.preprocessing import MinMaxScaler
s1 = MinMaxScaler(feature_range= (0,1))
s1.fit(X_train_num_engg)
X_train_num_scaled = s1.transform(X_train_num_engg)
X_val_num_scaled = s1.transform(X_val_num_engg)
print(X_train_num_scaled)

# Dimensionality Reduction : PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 9) # No of target dimensions
pca.fit(X_train_num_scaled)
X_train_num_trans = pca.transform(X_train_num_scaled)
X_val_num_trans = pca.transform(X_val_num_scaled)
print(X_train_num_trans.shape)
print(X_train_num_trans)

X_train_cat = X_train_cat.astype(str)
X_val_cat = X_val_cat.astype(str)
X_train_cat_trans = pd.get_dummies(X_train_cat)
X_val_cat_trans = pd.get_dummies(X_val_cat)
print(X_train_cat_trans.shape,X_val_cat_trans.shape)
print(X_train_cat_trans)

X_train_trans = np.concatenate((X_train_cat_trans, X_train_num_trans), axis=1)
X_val_trans = np.concatenate((X_val_cat_trans, X_val_num_trans), axis=1)
print(X_train_trans.shape)
print(X_val_trans.shape)

# Scaling Labels
from sklearn.preprocessing import MinMaxScaler
s2 = MinMaxScaler(feature_range= (0,1))
s2.fit(y_train.to_numpy().reshape(-1,1))
y_train_scaled = s2.transform(y_train.to_numpy().reshape(-1,1))
y_val_scaled = s2.transform(y_val.to_numpy().reshape(-1,1))
print(y_train_scaled.shape, y_val_scaled.shape)
print(y_train_scaled)

# Neural Network Model
model = ak.StructuredDataRegressor(overwrite=True, max_trials=10)
model.fit(X_train_trans,y_train_scaled, validation_data=(X_val_trans, y_val_scaled))
model = model.export_model()
model.summary()
# Saving Model
model.save('abalone')
#Loading Model
from tensorflow.keras.models import load_model
model = load_model('abalone')

# Prepare Test Dataset
X_test_num_engg = pol.transform(X_test_num)
X_test_num_scaled = s1.transform(X_test_num_engg)
X_test_num_trans = pca.transform(X_test_num_scaled)
X_test_num_trans = pd.DataFrame(X_test_num_trans)
X_test_cat = X_test_cat.astype(str)
X_test_cat_trans = pd.get_dummies(X_test_cat)
X_test_trans = np.concatenate((X_test_cat_trans, X_test_num_trans), axis=1)
y_test_scaled = s2.transform(y_test.to_numpy().reshape(-1,1))
print(X_test_trans.shape, y_test_scaled.shape)

# Evaluate model on Test Dataset
print(model.evaluate(X_test_trans, y_test_scaled))
y_pred = model.predict(X_test_trans)
y_pred = s2.inverse_transform(y_pred) # Predicts No of rings
y_pred = y_pred.flatten()
print(y_pred.shape)
print(y_pred)

mae = np.mean(np.abs(y_pred-y_test))
mse = np.mean(np.square(y_pred-y_test))
print('Mean Absolute Error:%.2f, Mean Square Error:%.2f' %(mae,mse))