#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from model_functions import GaussRankScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from sklearn.preprocessing import OrdinalEncoder
from numpy import asarray
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from yellowbrick.regressor import residuals_plot
import joblib
import sys
from matplotlib.patches import Patch
import itertools
from collections import deque
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import f1_score

#%%

sys.stdout = open("test.txt", "w")

plt.rcParams['font.family'] = "Times New Roman"

plt.rcParams['axes.linewidth']=1
plt.rcParams['axes.edgecolor']='black'
plt.rcParams["font.size"] = "20"

print('')
print('Estimating oil rcovery factor using XGBoost.')
print('')
print('Importing the database')
print('')

df = pd.read_csv("commercial_atlas.csv", encoding = "ISO-8859-1",
                 engine = 'python')

dftoris = pd.read_csv("toris.csv", encoding = "ISO-8859-1",
                 engine = 'python')

dftoris = dftoris.dropna(how='any')

#%%

print('Cleaning the database')
print('')

df1 = df.sample(frac=1, random_state = 2)

df1 = df1.dropna(how='any', subset=['RECOVERY FACTOR (OIL ULTIMATE)'])

df1 = df1[(df1['RECOVERY FACTOR'] >= 0) | (df1['RECOVERY FACTOR'].isnull())]

df1 = df1[(df1['RECOVERY FACTOR'] <= 1) | (df1['RECOVERY FACTOR'].isnull())]

df1 = df1[(df1['POROSITY'] >= 0) | (df1['POROSITY'].isnull())]

df1 = df1[(df1['POROSITY'] <= 1) | (df1['POROSITY'].isnull())]

df1 = df1[(df1['WATER SATURATION'] >= 0) | (df1['WATER SATURATION'].isnull())]

df1 = df1[(df1['WATER SATURATION'] <= 1) | (df1['WATER SATURATION'].isnull())]

df1 = df1[(df1['FVF'] >= 0) | (df1['FVF'].isnull())]

df1 = df1[(df1['FVF'] <= 10) | (df1['FVF'].isnull())]

df1 = df1[(df1['GOR'] >= 0) | (df1['GOR)'].isnull())]

df1 = df1[(df1['GOR'] <= 60) | (df1['GOR'].isnull())]

df1 = df1[(df1['RESERVES'] >= 0) | (df1['RESERVES'].isnull())]

df1 = df1[(df1['RESERVES'] <= 5e+11) | (df1['RESERVES'].isnull())]

clt = round(0.55*df1.shape[1])

rwt = round(0.70*df1.shape[0])

df1 = df1.dropna(axis=0, thresh=clt)

df1 = df1.dropna(axis=1, thresh=rwt)

df1 = df1.sort_values(by ='RECOVERY FACTOR (OIL ULTIMATE)',ascending=True)

df1 = df1.reset_index(drop=True)

dfnum = df1.select_dtypes(include=['number'])
    
#%%

dfcat = df1.select_dtypes(include=['object'])

dfcat = dfcat.astype('category')

dfcat = dfcat.fillna(dfcat.mode().iloc[0])

#%%

df2 = pd.concat([dfnum, dfcat], axis = 1)

#%%

df_x = df2.iloc[:,:-1]

df_y = df2.iloc[:,-1]

df_y = asarray(df_y)

df_y = df_y.reshape(-1, 1)

del df_x["RECOVERY FACTOR (OIL ULTIMATE)"]

encoder = OrdinalEncoder()

df_y = encoder.fit_transform(df_y)

X_train, X_test, y_train, y_test= train_test_split(df_x, df_y,
                                                   test_size=0.1,
                                                   random_state=42,
                                                   stratify = df_y)

y_train = y_train.astype(int)

y_test = y_test.astype(int)

print("imputing for the missing data.")

t = 0.10

n = 10

def mode_calculator(df):
    mo=df.mode()
    if mo.shape[0] == 1:
        zmo=mo[0]
    else:
        zmo=mo.mean(axis=0)
    return zmo

def missing_value_handler(i_start, i_end, df):
    temp = df[i_start:i_end]
    fr = temp.isnull().sum()/temp.shape[0]
    if fr > t:
        while fr > t:
            if df.shape[0] - i_end != 0:
                temp = df[i_start:i_end+1]
                i_end += 1
                fr = temp.isnull().sum()/temp.shape[0]
            zmo = mode_calculator(temp)
            temp = temp.replace(np.nan, zmo)
            return temp
        else:
                zmo = mode_calculator(temp)
                temp = temp.replace(np.nan, zmo)
                return temp
    elif fr <= t:
        zmo = mode_calculator(temp)
        temp = temp.replace(np.nan, zmo)
        return temp
    elif fr == 0:
        return temp

for col in list(X_train.columns.values):
    my_col = X_train[col]
    temp_df = pd.DataFrame()
    i_start = 0
    i_end = n
    
    while my_col.shape[0] - i_start >= n:
            curated_temp = missing_value_handler(i_start, i_end, my_col)
            temp_df = temp_df.append(curated_temp.to_frame())
            i_start = i_start+curated_temp.shape[0]
            i_end = i_start + n
            
    last_piece = missing_value_handler(i_start, my_col.shape[0], my_col)
    if last_piece is None:
        X_train[col+"_curated"] = temp_df
        X_train=X_train.drop([col], axis=1)

    else:
        temp_df = temp_df.append(last_piece.to_frame())
        X_train[col+"_curated"] = temp_df
        X_train=X_train.drop([col], axis=1)
        
pass

for col in list(X_train.columns.values):
    X_train[col] = X_train[col].fillna(mode_calculator(X_train[col]))
    
pass



for col in list(X_test.columns.values):
    my_col = X_test[col]
    temp_df = pd.DataFrame()
    i_start = 0
    i_end = n
    
    while my_col.shape[0] - i_start >= n:
            curated_temp = missing_value_handler(i_start, i_end, my_col)
            temp_df = temp_df.append(curated_temp.to_frame())
            i_start = i_start+curated_temp.shape[0]
            i_end = i_start + n
            
    last_piece = missing_value_handler(i_start, my_col.shape[0], my_col)
    if last_piece is None:
        X_test[col+"_curated"] = temp_df
        X_test=X_test.drop([col], axis=1)

    else:
        temp_df = temp_df.append(last_piece.to_frame())
        X_test[col+"_curated"] = temp_df
        X_test=X_test.drop([col], axis=1)
        
pass

for col in list(X_train.columns.values):
    X_test[col] = X_test[col].fillna(mode_calculator(X_test[col]))
    
pass

gauss_scaler = GaussRankScaler()

X_trainnum = gauss_scaler.fit_transform(X_train.iloc[:, -11:])

X_testnum = gauss_scaler.transform(X_test.iloc[:, -11:])

X_trainnum_columns = len(X_trainnum[0])

num_column_names = X_train.iloc[:, -11:].columns.values.tolist()

#%%

scaler = preprocessing.MinMaxScaler()
    
X_trainnum = scaler.fit_transform(X_trainnum)

X_testnum = scaler.transform(X_testnum)

#%%

test_accu = pd.DataFrame(columns = ['Test', 'accuracy'])

train_accu = pd.DataFrame(columns = ['Train', 'accuracy'])

toris_accu = pd.DataFrame(columns = ['Train', 'accuracy'])

y_test_values = pd.DataFrame(columns = ['y_measured', 'y_estimated'])

y_train_values = pd.DataFrame(columns = ['y_measured', 'y_estimated'])

y_predtoris_values = pd.DataFrame(columns = ['y_measured', 'y_estimated'])

#%%

num_boost_round = 999

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names = df_x.columns)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names = df_x.columns)

params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'booster': 'gbtree',
    'nthread': 10,
    'validate_parameters':'True',
    'alpha': 0.2,
    'lambda': 0.001,
    'colsample_bylevel': 0.9,
    'verbose': 0,
    'gamma': 0.01,
    'max_delta_step': 0.1,
    'silent': 0,
    'num_class': 10,
}


min_mlogloss = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print('')
    print("CV with eta={}".format(eta))
    print('') 
    params['eta'] = eta
    
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics=['mlogloss'],
            early_stopping_rounds=10
          )

    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print('')
    print("\tmlogloss {} for {} rounds\n".format(mean_mlogloss, boost_rounds))
    print('')
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = eta

print('')
print("Best parameter: eta = {}, mlogloss: {}".format(best_params, min_mlogloss))
print('')

params['eta'] = best_params



gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(1,12)
    for min_child_weight in range(1,11)
]

min_mlogloss = float("Inf")
best_params = None

for max_depth, min_child_weight in gridsearch_params:
    print('')
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    print('')
    
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=10,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )

    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print('')
    print("\tmlogloss {} for {} rounds".format(mean_mlogloss, boost_rounds))
    print('')
    
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (max_depth,min_child_weight)
        
print('')
print("Best parameters: max_depth = {}, min_child_weight = {}, mlogloss: {}".format(best_params[0], best_params[1], min_mlogloss))
print('')

params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]



gridsearch_params = [
    (subsample, colsample_bytree)
    for subsample in [i/10. for i in range(1,11)]
    for colsample_bytree in [i/10. for i in range(1,11)]
]

min_mlogloss = float("Inf")
best_params = None

for subsample, colsample_bytree in reversed(gridsearch_params):
    print('')
    print("CV with subsample={}, colsample_bytree={}".format(
                             subsample,
                             colsample_bytree))
    print('')
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample_bytree
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=10,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )

    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print('')
    print("\tmlogloss {} for {} rounds".format(mean_mlogloss, boost_rounds))
    print('')
    
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (subsample,colsample_bytree)

print('')
print("Best params: {}, {}, mlogloss: {}".format(best_params[0], best_params[1],
                                             min_mlogloss))
print('')

params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]



gridsearch_params = [
    (max_delta_step, colsample_bylevel)
    for max_delta_step in [i/10. for i in range(1,11)]
    for colsample_bylevel in [i/10. for i in range(1,11)]
]

min_mlogloss = float("Inf")
best_params = None

for max_delta_step, colsample_bylevel in reversed(gridsearch_params):
    print('')
    print("CV with max_delta_step={}, colsample_bylevel={}".format(
                             max_delta_step,
                             colsample_bylevel))
    print('')
    params['max_delta_step'] = max_delta_step
    params['colsample_bylevel'] = colsample_bylevel
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=10,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )

    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print('')
    print("\tmlogloss {} for {} rounds".format(mean_mlogloss, boost_rounds))
    print('')
    
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (max_delta_step,colsample_bylevel)

print('')
print("Best params: {}, {}, mlogloss: {}".format(best_params[0], best_params[1],
                                             min_mlogloss))
print('')

params['max_delta_step'] = best_params[0]
params['colsample_bylevel'] = best_params[1]



gridsearch_params = [
    (ralpha, rlambda)
    for ralpha in [i/10. for i in range(1,10)]
    for rlambda in [i/100. for i in range(1,10)]
]

min_mlogloss = float("Inf")
best_params = None

for ralpha, rlambda in reversed(gridsearch_params):
    print('')
    print("CV with alpha={}, lambda={}".format(
                             ralpha,
                             rlambda))
    print('')
    params['alpha'] = ralpha
    params['lambda'] = rlambda
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=10,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )

    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print('')
    print("\tmlogloss {} for {} rounds".format(mean_mlogloss, boost_rounds))
    print('')
    
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (ralpha,rlambda)

print('')
print("Best params: {}, {}, mlogloss: {}".format(best_params[0], best_params[1],
                                             min_mlogloss))
print('')

params['alpha'] = best_params[0]
params['lambda'] = best_params[1]



gridsearch_params = [
    (gamma)
    for gamma in [i/100. for i in range(1,10)]
]

min_mlogloss = float("Inf")
best_params = None

for gamma in reversed(gridsearch_params):
    print('')
    print("CV with gamma={}".format(
                             gamma))
    print('')
    params['gamma'] = gamma
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=10,
        metrics={'mlogloss'},
        early_stopping_rounds=10
    )

    mean_mlogloss = cv_results['test-mlogloss-mean'].min()
    boost_rounds = cv_results['test-mlogloss-mean'].argmin()
    print('')
    print("\tmlogloss {} for {} rounds".format(mean_mlogloss, boost_rounds))
    print('')
    
    if mean_mlogloss < min_mlogloss:
        min_mlogloss = mean_mlogloss
        best_params = (gamma)

print('')
print("Best params: {}, {}".format(best_params, min_mlogloss))
print('')

params['gamma'] = best_params



print('Fitting the model')

best_model = XGBClassifier(**params,early_stopping_rounds=10,num_boost_round=999, 
                           use_label_encoder = False)

best_model.fit(X_trainnum, y_train)

joblib.dump(best_model,'best_model_commercial_atlas_classified')

y_pred = best_model.predict(X_testnum)

y_pred1 = best_model.predict(X_trainnum)

score=accuracy_score(y_test, y_pred)
score1=accuracy_score(y_train, y_pred1)

macro_averaged_f1 = f1_score(y_test, y_pred, average = 'macro')
print(f"Macro-Averaged F1 score using sklearn library : {macro_averaged_f1}")

macro_averaged_f1_train = f1_score(y_train, y_pred1, average = 'macro')
print(f"Macro-Averaged F1 score using sklearn library : {macro_averaged_f1_train}")

print('')
print("Test accuracy is equal to", score)
print('')
print("Train accuracy is equal to", score1)
print('')

plt.figure(figsize=(8,4))
plt.scatter(y_test, y_pred, c='r', label='Test Measured RF')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.plot([0, 10], [0, 10], color = 'black', linewidth = 1, label="y = x")
plt.xlabel('Measured RF')
plt.ylabel('Esitmated RF')
plt.title('Measured RF Vs Estimated RF (Test Dataset)')
plt.legend()
plt.grid(False)
plt.savefig('Measured RF Vs Estimated RF (Test Dataset)',
            bbox_inches='tight')

plt.figure(figsize=(8,4))
plt.scatter(y_train, y_pred1, c='b', label='Train Measured RF')
plt.plot([0, 10], [0, 10], color = 'black', linewidth = 1, label="y = x")
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel('Measured RF')
plt.ylabel('Esitmated RF')
plt.title('Measured RF Vs Estimated RF (Train Dataset)')
plt.legend()
plt.grid(False)
plt.savefig('Measured RF Vs Estimated RF (Train Dataset)',
            bbox_inches='tight')

plt.figure(figsize=(8,4))
x_ax = range(len(y_test))
plt.plot(x_ax, y_test,'.', label="Measured", color="orange")
plt.plot(x_ax, y_pred,'.', label="Estimated", color="blue")
plt.xlabel('Sample')
plt.ylabel('Measured/Estimated RF')
plt.title("Measured RF Vs. Estimated RF Test Dataset")
plt.legend()
plt.grid(False)
plt.savefig('Measured Ksat Vs. Estimated Ksat (Test Dataset).png',
            bbox_inches='tight')

plt.figure(figsize=(8,4))
x_ax = range(len(y_train))
plt.plot(x_ax, y_train,'.', label="Measured", color="orange")
plt.plot(x_ax, y_pred1,'.', label="Estimated", color="blue")
plt.xlabel('Sample')
plt.ylabel('Measured/Estimated RF')
plt.title("Measured RF Vs. Estimated RF Train Dataset")
plt.legend()
plt.grid(False)
plt.savefig('Measured Ksat Vs. Estimated Ksat (Train Dataset).png',
            bbox_inches='tight')


#%% feature importance


print('Feature Importance of the models fitted on the commercial Atlas database')

explainer = shap.TreeExplainer(best_model, feature_perturbation = 'tree_path_dependent')

shap_values = explainer.shap_values(X_trainnum, y_train)

plt.figure(figsize=(10,6))
plt.title('Database CA')
shap.summary_plot(shap_values, X_trainnum, plot_type = "bar", 
                  feature_names = df_x.columns, show = False)
plt.grid(False)
plt.savefig('commercial_AtlasXGBC_average_impact_on_model_output_magnitude.png', bbox_inches='tight',facecolor='w',dpi=1200)

plt.figure(figsize=(10,6))
#plt.title('commercial Atlas ModelXGBC Feature Importance')
shap.summary_plot(shap_values[0], X_trainnum, feature_names = df_x.columns,
                  show = False)
plt.grid(False)
plt.savefig('commercial_AtlasXGBR_impact_on_model_output.png', bbox_inches='tight',facecolor='w',dpi=1200)


# for col in range(0, X_trainnum_columns, 1):
#     plt.figure(figsize=(10,6))
#     plt.title('commercial Atlas ModelXGBR Dependency Plots')
#     shap.dependence_plot(col,shap_values, X_trainnum, feature_names=df_x.columns)
#     plt.grid(False)
#     plt.savefig('commercial_AtlasXGBR_dependency_plot_{}.png'.format(df_x.columns([col])), bbox_inches='tight',facecolor='w')

#%%     

print("TORIS data clean up.")

dftoris = dftoris[(dftoris['RECOVERY FACTOR'] >= 0) | (dftoris['RECOVERY FACTOR'].isnull())]

dftoris = dftoris[(dftoris['RECOVERY FACTOR'] <= 1) | (dftoris['RECOVERY FACTOR'].isnull())]

dftoris = dftoris[(dftoris['POROSITY'] >= 0) | (dftoris['POROSITY'].isnull())]

dftoris = dftoris[(dftoris['POROSITY'] <= 1) | (dftoris['POROSITY'].isnull())]

dftoris = dftoris[(dftoris['WATER SATURATION'] >= 0) | (dftoris['WATER SATURATION'].isnull())]

dftoris = dftoris[(dftoris['WATER SATURATION'] <= 1) | (dftoris['WATER SATURATION'].isnull())]

dftoris = dftoris[(dftoris['FVF'] >= 0) | (dftoris['FVF'].isnull())]

dftoris = dftoris[(dftoris['FVF'] <= 10) | (dftoris['FVF'].isnull())]

dftoris = dftoris[(dftoris['GOR'] >= 0) | (dftoris['GOR)'].isnull())]

dftoris = dftoris[(dftoris['GOR'] <= 60) | (dftoris['GOR'].isnull())]

dftoris = dftoris[(dftoris['RESERVES'] >= 0) | (dftoris['RESERVES'].isnull())]

dftoris = dftoris[(dftoris['RESERVES'] <= 5e+11) | (dftoris['RESERVES'].isnull())]
print('TORIS data processing.')

Xtoris = dftoris.iloc[:, :-1]

del Xtoris["RECOVERY FACTOR (OIL ULTIMATE)"]

Xtoris = gauss_scaler.fit_transform(Xtoris)

Xtoris = scaler.fit_transform(Xtoris)

ytoris = dftoris.iloc[:, -1]

encoder = OrdinalEncoder()

ytoris = ytoris.values.reshape(-1, 1)

ytoris = encoder.fit_transform(ytoris)

ytoris = ytoris.astype(int)

bestmodel = joblib.load('best_model_commercial_atlas_classified')

y_predtoris = bestmodel.predict(Xtoris)

scoretoris=accuracy_score(ytoris, y_predtoris)

print('')
print("commercial accuracy is equal to", scoretoris)

macro_averaged_f1_toris = f1_score(ytoris, y_predtoris, average = 'macro')
print(f"Macro-Averaged F1 score using sklearn library : {macro_averaged_f1_toris}")

y_test=y_test.flatten()

y_train=y_train.flatten()

ytoris=ytoris.flatten()


#%%

y_test_values = pd.concat([y_test_values,pd.DataFrame({"y_measured": y_test,
                            "y_estimated": y_pred})], ignore_index = True)
        
y_train_values = pd.concat([y_train_values,
                            pd.DataFrame({"y_measured": y_train,
                                          "y_estimated": y_pred1})],
                           ignore_index = True)

test_score = test_accu.append({"Test": "Test", "accuracy_value" : score},
                             ignore_index = True)

train_score = train_accu.append({"Train": "Train", "accuracy_value" : score1},
                             ignore_index = True)

toris_score = toris_accu.append({"Train": "Train", "accuracy_value" : scoretoris},
                             ignore_index = True)

best_parameters = pd.DataFrame.from_dict(params, orient='index').rename(
    columns={0: 'Value'})

x_set = pd.DataFrame(data = X_trainnum, columns = df_x.columns)

x_settest = pd.DataFrame(data = X_testnum, columns = df_x.columns)

x_settoris = pd.DataFrame(data = Xtoris, columns = df_x.columns)

y_predtoris_values = pd.concat([y_predtoris_values,pd.DataFrame({"y_measured": ytoris,
                            "y_estimated": y_predtoris})], ignore_index = True)

with pd.ExcelWriter('model_data.xlsx', engine="openpyxl") as writer:
    
    y_test_values.to_excel(writer, sheet_name = 'y_test_values')
        
    y_train_values.to_excel(writer, sheet_name = 'y_train_values')
    
    y_predtoris_values.to_excel(writer, sheet_name = 'y_predtoris_values')
            
    test_score.to_excel(writer, sheet_name = 'test_accuracy')
            
    train_score.to_excel(writer, sheet_name = 'train_accuracy')
    
    toris_score.to_excel(writer, sheet_name = 'toris_accuracy')
    
    best_parameters.to_excel(writer, sheet_name = 'best parameters')
    
    x_set.to_excel(writer, sheet_name = 'x train')
    
    x_settest.to_excel(writer, sheet_name = 'x test')
    
    x_settoris.to_excel(writer, sheet_name = 'x toris')
    
    
import pandas as pd
from scipy.stats import ks_2samp
datatrainres = X_train.iloc[:,7]
datatestres = X_test.iloc[:,7]
dataindres=dftoris.iloc[:,7]
testres = ks_2samp(datatrainres,datatestres)
testindres = ks_2samp(datatrainres,dataindres)

datatrainarea = X_train.iloc[:,10]
datatestarea = X_test.iloc[:,10]
dataindarea=dftoris.iloc[:,10]
testarea = ks_2samp(datatrainarea,datatestarea)
testindarea = ks_2samp(datatrainarea,dataindarea)

datatrainthic = X_train.iloc[:,6]
datatestthic = X_test.iloc[:,6]
dataindthic=dftoris.iloc[:,6]
testthic = ks_2samp(datatrainthic,datatestthic)
testindthic = ks_2samp(datatrainthic,dataindthic)

y_train1=pd.DataFrame(y_train)
y_train1=y_train1.squeeze()
y_test1=pd.DataFrame(y_test)
y_test1=y_test1.squeeze()
yind=dftoris.iloc[:,11]
testrf = ks_2samp(y_train1,y_test1)
testindrf = ks_2samp(y_train1,yind)

