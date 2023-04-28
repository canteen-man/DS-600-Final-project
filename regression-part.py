import numpy as np
import pandas as pd
import re
import datetime as dt
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

## Regression using scikit-learn
from sklearn.linear_model import Lasso, ElasticNet

from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder, scale, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score


####################################################load data####################################################


#input threshold
print('Enter correlation value(from 0.1 to 0.3):')
threshold = float(input())
print('threshold,', threshold)

data = pd.read_csv("./new.csv", parse_dates=["tradeTime"], encoding='gb2312')

# Remove the "url" and "id" "Cid" columns
data= data.drop(["url", "id","Cid","price"], axis=1)

# Check the data
data.info()

####################################################deal with object values/data types####################################################

##--tradetime column
data['tradeTime'] = pd.to_datetime(data['tradeTime'])
data['tradeYear'] = data['tradeTime'].dt.year
data['tradeMonth'] = data['tradeTime'].dt.month
data['tradeDay'] = data['tradeTime'].dt.day
# Drop the original 'tradeTime' column
data = data.drop('tradeTime', axis=1)

##--constructionTime column
#see how many foreign characters in this column
data['constructionTime'].value_counts()
# Remove rows with non-ASCII characters in the 'constructionTime' column
data = data[data['constructionTime'].apply(lambda x: x.isascii())]

# Check the value counts after cleaning
print("\nValue counts after cleaning:")
data['constructionTime'].value_counts()



##--floor column

# Define a function to extract numbers from a string
def extract_numbers(s):
    return ''.join(re.findall(r'\d+', s))

# Apply the function to the 'floor' column
data['floor'] = data['floor'].apply(extract_numbers)

# Filter out rows with empty strings in the 'floor' column
data = data[data['floor'] != '']
print(data['floor'].value_counts())


# changing data types

data = data.astype({
    'floor': 'int32',
    'constructionTime': 'int32',
    'livingRoom': 'int32',
    'drawingRoom':'int32',
    'bathRoom': 'int32'
})
# Check the data
data.info()


# Identify missing values
data.isna().sum()
##--DOM is a major one
plt.clf()
sns.boxplot(y = data["DOM"])
plt.xlabel("DOM")
plt.ylabel("distribution")
plt.show()

plt.clf()
sns.kdeplot(data = data.loc[:,"DOM"])
plt.show()

data["DOM"].quantile([0.25,0.5,0.75,1])

#fill NA with median
data["DOM"] = data["DOM"].fillna(data["DOM"].median())
# Removing the other records from the dataset having missing values


##--communityAverage
#fill NA with mean
data["communityAverage"] = data["communityAverage"].fillna(data["communityAverage"].median())

#all other NA
data.dropna(axis = 0, how = "any", inplace = True)


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation heatmap for the entire dataset
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap for All Features")
plt.show()

# Select the features that have a correlation above the threshold (ignoring the 'totalPrice' itself)
totalprice_corr = correlation_matrix['totalPrice']
selected_features = totalprice_corr[abs(totalprice_corr) > threshold].drop('totalPrice').index

# Create a cleaned dataset with only the selected features and 'totalPrice' column
cleaned_data = data[selected_features.union(['totalPrice'])]

# Check the cleaned_data DataFrame
print(cleaned_data.head())

# Identify missing values
cleaned_data.isna().sum()



####################################################Regression####################################################

df = cleaned_data
df = shuffle(df)
X = df.drop('totalPrice', axis = 1).to_numpy()
y = df['totalPrice'].to_numpy()

results_df = pd.DataFrame()
columns = ["Model", "MAE", "MSE", "RMSE", "R2"]

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def append_results(model_name, model, results_df, y_test, pred):
    results_append_df = pd.DataFrame(data=[[model_name, *evaluate(y_test, pred)]], columns=columns)
    results_df = results_df.append(results_append_df, ignore_index = True)
    return results_df

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA()
pca.fit(X_std)
top_5_components = pca.components_[:5, :]
X_std_transformed = X_std.dot(top_5_components.T)

X_pca_train = X_std_transformed[:230000,:]
X_pca_test = X_std_transformed[230000:,:]
X_train = X[:230000,:]
X_test = X[230000:,:]
y_train = y[:230000]
y_test =y[230000:]

X_pca_train_csv = pd.DataFrame(X_pca_train)
X_pca_test_csv = pd.DataFrame(X_pca_test)
X_train_csv = pd.DataFrame(X_train)
X_test_csv = pd.DataFrame(X_test)
y_train_csv = pd.DataFrame(y_train)
y_test_csv = pd.DataFrame(y_test)

X_pca_train_csv.to_csv('./X_pca_train.csv', index=False)
X_pca_test_csv.to_csv('./X_pca_test.csv', index=False)
X_train_csv.to_csv('./X_train.csv', index=False)
X_test_csv.to_csv('./X_test.csv', index=False)
y_train_csv.to_csv('./y_train.csv', index=False)
y_test_csv.to_csv('./y_test.csv', index=False)




#############################################lasso regression#########################################
model = Lasso()
#####################################original lasso regression#########################################
model.fit(X_train, y_train)
pred = model.predict(X_test)
results_df = append_results("Original Lasso Regression",  Lasso(), results_df, y_test, pred)

#####################################PCA lasso regression#########################################
model.fit(X_pca_train, y_train)
pred = model.predict(X_pca_test)
results_df = append_results("PCA Lasso Regression",  Lasso(), results_df, y_test, pred)

#####################RFE lasso regression#########################################
rfe = RFE(estimator=model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X, y)
model.fit(X_rfe[:230000,:], y_train)
pred = model.predict(X_rfe[230000:,:])
results_df = append_results("RFE Lasso Regression", Lasso(), results_df, y_test, pred)



#####################################ElasticNet regression#########################################
model = ElasticNet()
#####################################original ElasticNet regression####################
model.fit(X_train, y_train)
pred = model.predict(X_test)
results_df = append_results("Original ElasticNet Regression",  ElasticNet(), results_df, y_test, pred)
#####################################PCA ElasticNet regression#############################
model.fit(X_pca_train, y_train)
pred = model.predict(X_pca_test)
results_df = append_results("PCA ElasticNet Regression",  ElasticNet(), results_df, y_test, pred)
#####################RFE lasso regression#########################################
rfe = RFE(estimator=model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X, y)
model.fit(X_rfe[:230000,:], y_train)
pred = model.predict(X_rfe[230000:,:])
results_df = append_results("RFE ElasticNet Regression", ElasticNet(), results_df, y_test, pred)

print("threshold is \n", threshold)
print(results_df.to_string())





