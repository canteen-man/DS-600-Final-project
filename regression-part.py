import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

#replace the new csv after data clean @jitong yu
df = pd.read_csv("./new.csv", encoding='iso-8859-1', low_memory = False)
df.head()
df = df[['totalPrice', 'square', 'renovationCondition', 'communityAverage','followers','elevator','ladderRatio' ]]
df.dropna(inplace=True)
df.info()
df = shuffle(df)

X = df.drop('totalPrice', axis = 1).to_numpy()
y = df['totalPrice'].to_numpy()


from sklearn import metrics
from sklearn.model_selection import cross_val_score

results_df = pd.DataFrame()
columns = ["Model", "Cross Val Score", "MAE", "MSE", "RMSE", "R2"]

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def append_results(model_name, model, results_df, y_test, pred):
    results_append_df = pd.DataFrame(data=[[model_name, *evaluate(y_test, pred) , cross_val_score(model, X, y, cv=10).mean()]], columns=columns)
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
from sklearn.linear_model import Lasso
model = Lasso()
#####################################original lasso regression#########################################
model.fit(X_train, y_train)
pred = model.predict(X_test)
results_df = append_results("Original Lasso Regression",  Lasso(), results_df, y_test, pred)

#####################################PCA lasso regression#########################################
model.fit(X_pca_train, y_train)
pred = model.predict(X_pca_test)
results_df = append_results("PCA Lasso Regression",  Lasso(), results_df, y_test, pred)



#############################################ridge regression#########################################
from sklearn.linear_model import Ridge

model = Ridge(normalize=True)
#####################################original ridge regression#########################################
model.fit(X_train, y_train)

pred = model.predict(X_test)
results_df = append_results("Original Ridge Regression",  Ridge(), results_df, y_test, pred)
#####################################PCA ridge regression#########################################
model.fit(X_pca_train, y_train)
pred = model.predict(X_pca_test)
results_df = append_results("PCA Ridge Regression",  Ridge(), results_df, y_test, pred)

results_df
