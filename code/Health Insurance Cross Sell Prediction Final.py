
# ### Import Packages
import statsmodels.api as sm
from statsmodels.formula.api import glm
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")


# ### Import Dataset

train = pd.read_csv('Project2_Train.csv')
test = pd.read_csv('Project2_Test.csv')


# ## 1. Logistic Regression

# #### Preprocessing Data

train_lg = pd.get_dummies(train, columns = ['Region_Code'])
train_lg = train_lg.drop(["id", "Region_Code_50"], axis = 1)
train_lg['Gender']=train_lg['Gender'].replace({'Male':1,'Female':0})
train_lg['Vehicle_Damage'].replace({'Yes':1,'No':0}, inplace=True)
train_lg['Vehicle_Age'].replace({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3}, inplace=True)
train_lg['Age']=np.log(train_lg['Age'])
train_lg['Annual_Premium']=np.log(train_lg['Annual_Premium'])
train_lg['Vintage']=np.log(train_lg['Vintage'])

X_lg = train_lg.loc[:, train_lg.columns != 'Response']
y_lg = np.ravel(train_lg.loc[:, train_lg.columns == 'Response'])
X_train_lg, X_val_lg, y_train_lg, y_val_lg = train_test_split(X_lg, y_lg, random_state = 42)


# #### Fitting the Model


logreg = LogisticRegressionCV(cv=5, scoring='accuracy')
logreg.fit(X_train_lg, y_train_lg)
y_pred_problg = logreg.predict_proba(X_val_lg)[:, 1]
probabilitythresholdlg = 0.25
y_pred_thresholdlg = (y_pred_problg >= probabilitythresholdlg).astype(int)
print('Accuracy: ',metrics.accuracy_score(y_val_lg, y_pred_thresholdlg))
print(classification_report(y_val_lg, y_pred_thresholdlg))


# #### Test for Overfitting

print('Score:', logreg.score(X_train_lg, y_train_lg))
print('Score:', logreg.score(X_val_lg, y_val_lg))


# #### AUC ###


fpr, tpr, threshold = metrics.roc_curve(y_val_lg, y_pred_problg)
roc_auc = metrics.auc(fpr, tpr)
auc_lr = metrics.roc_auc_score(y_val_lg, y_pred_problg)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### Model Summary 

logit_model = sm.Logit(y_lg, X_lg)
result = logit_model.fit()
print(result.summary())


# ## 2. Classification Tree: Gini

# #### Preprocessing Data


train_clf = pd.get_dummies(train, columns = ['Region_Code'])
train_clf = train_clf.drop(["id", "Region_Code_50"], axis = 1)
train_clf['Gender']=train_clf['Gender'].replace({'Male':1,'Female':0})
train_clf['Vehicle_Damage'].replace({'Yes':1,'No':0}, inplace=True)
train_clf['Vehicle_Age'].replace({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3}, inplace=True)
train_clf['Age']=np.log(train_clf['Age'])
train_clf['Annual_Premium']=np.log(train_clf['Annual_Premium'])
train_clf['Vintage']=np.log(train_clf['Vintage'])


X_clf = train_clf.loc[:, train_clf.columns != 'Response']
y_clf = np.ravel(train_clf.loc[:, train_clf.columns == 'Response'])
X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(X_clf, y_clf, random_state = 42)


# #### Hyperparameter Tuning 


DecisionTreeGini = DecisionTreeClassifier(criterion = "gini", random_state=0) 

# Setup the parameters and distributions to sample from:
param_dist = {"max_depth": np.arange(1,10),
              'min_samples_leaf': [2,4,6,8,10,12,14],
              'max_features': np.arange(1,10)}
                                
# Instantiate the GridSearchCV object:
clf_gridsearch = GridSearchCV(DecisionTreeGini, param_dist, scoring='roc_auc', cv=10)
clf_gridsearch.fit(X_train_clf, y_train_clf) 

best_hyperparams = clf_gridsearch.best_params_
print('Best hyerparameters:\n', best_hyperparams)


# #### Fitting the Model


DecisionTreeCLF = DecisionTreeClassifier(criterion = "gini", random_state=0, max_depth = 4, max_features = 8, min_samples_leaf = 12)
DecisionTreeCLF.fit(X_train_clf,y_train_clf)
y_pred_probclf = DecisionTreeCLF.predict_proba(X_val_clf)[:, 1]
probabilitythreshold_clf=0.25
y_pred_thresholdclf = (y_pred_probclf>=probabilitythreshold_clf).astype(int)

confusion_matrix_clf = pd.crosstab(y_val_clf, y_pred_thresholdclf, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_clf, annot=True, fmt='')

print('Accuracy: ',metrics.accuracy_score(y_val_clf, y_pred_thresholdclf))
print(classification_report(y_val_clf, y_pred_thresholdclf))


# #### AUC


fpr, tpr, threshold = metrics.roc_curve(y_val_clf, y_pred_probclf)
roc_auc = metrics.auc(fpr, tpr)
auc_clf = metrics.roc_auc_score(y_val_clf, y_pred_probclf)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### Plot Classification tree diagram

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=500)
tree.plot_tree(DecisionTreeCLF,
               filled = True);
fig.savefig('imagename.png')


# ## 3. Random Forest: Gini 

# #### Preprocessing Data

train_rf = pd.get_dummies(train, columns = ['Region_Code'])
train_rf = train_rf.drop(["id", "Region_Code_50"], axis = 1)
train_rf['Gender']=train_rf['Gender'].replace({'Male':1,'Female':0})
train_rf['Vehicle_Damage'].replace({'Yes':1,'No':0}, inplace=True)
train_rf['Vehicle_Age'].replace({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3}, inplace=True)
train_rf['Age']=np.log(train_rf['Age'])
train_rf['Annual_Premium']=np.log(train_rf['Annual_Premium'])
train_rf['Vintage']=np.log(train_rf['Vintage'])



X_rf = train_rf.loc[:, train_rf.columns != 'Response']
y_rf = np.ravel(train_rf.loc[:, train_rf.columns == 'Response'])
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X_rf, y_rf, random_state = 0)


# #### Hyperparameter Tuning
# 



RandomForest = RandomForestClassifier(criterion='gini', random_state=42)

# Setup the parameters and distributions to sample from:
param_dist = {'max_depth': np.arange(3,10),
              'min_samples_leaf': [1,2,4,6,8],
              'max_features': [1,2,4,6,8],
              'n_estimators': [100,200,300]
             }

# Instantiate the GridSearchCV object: 
RF_cv = GridSearchCV(RandomForest, param_dist,scoring='roc_auc', cv=5)
RF_cv.fit(X_train_rf,y_train_rf)


best_hyperparams = RF_cv.best_params_
print('Best hyerparameters:\n', best_hyperparams)


# #### Fitting the Model


RandomForestCLF = RandomForestClassifier(criterion='gini', n_estimators = 300, max_features=8, max_depth = 5, random_state = 42, min_samples_leaf= 8)
RandomForestCLF.fit(X_train_rf,y_train_rf)
y_pred_probrf = RandomForestCLF.predict_proba(X_val_rf)[:, 1]
probabilitythreshold_rf = 0.25
y_pred_thresholdrf = (y_pred_probrf>=probabilitythreshold_rf).astype(int)

confusion_matrix_rf = pd.crosstab(y_val_rf, y_pred_thresholdrf, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_rf, annot=True, fmt='g')

print('Accuracy: ',metrics.accuracy_score(y_val_rf, y_pred_thresholdrf))
print(classification_report(y_val_rf, y_pred_thresholdrf))


# #### Feature importance



feature_imp_rf = pd.Series(RandomForestCLF.feature_importances_,index=X_rf.columns.values).sort_values(ascending=False)
feature_imp_rf


# #### AUC



fpr, tpr, threshold = metrics.roc_curve(y_val_rf, y_pred_probrf)
roc_auc = metrics.auc(fpr, tpr)
auc_rf = metrics.roc_auc_score(y_val_rf, y_pred_probrf)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## 4. LightGBM

# #### Preprocessing Data




train_gbm = pd.get_dummies(train, columns = ['Region_Code'])
train_gbm = train_gbm.drop(["id", "Region_Code_50"], axis = 1)
train_gbm['Gender']=train_rf['Gender'].replace({'Male':1,'Female':0})
train_gbm['Vehicle_Damage'].replace({'Yes':1,'No':0}, inplace=True)
train_gbm['Vehicle_Age'].replace({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3}, inplace=True)
train_gbm['Age']=np.log(train_rf['Age'])
train_gbm['Annual_Premium']=np.log(train_gbm['Annual_Premium'])
train_gbm['Vintage']=np.log(train_gbm['Vintage'])



y = train_gbm.Response
X = train_gbm.drop('Response', axis =1)

X_train_gbm, X_val_gbm, y_train_gbm, y_val_gbm = train_test_split(X, y, test_size = 0.3, random_state = 0)


# #### Hyperparameter Tuning



lgbm_clf = LGBMClassifier()

# Setup the parameters and distributions to sample from:
param_dist = {'max_depth':[2,3,4,5],
              'n_estimators': [250, 300, 350],
              'num_leaves' : [4,5,6,7],
              'learning_rate' : [0.1,0.01, 0.001]
             }

# Instantiate the GridSearchCV object: 
lgbm_cv = GridSearchCV(lgbm_clf, param_dist,scoring='roc_auc', cv=5)
lgbm_cv.fit(X_train_gbm,y_train_gbm)


best_hyperparams = lgbm_cv.best_params_
print('Best hyerparameters:\n', best_hyperparams)


# #### Fitting the Model


lgbm_clf = LGBMClassifier( max_depth = 3, n_estimators = 350, num_leaves = 4, learning_rate = 0.01)
lgbm_clf.fit(X_train_gbm, y_train_gbm)
lgbm_clf_pred_proba = lgbm_clf.predict_proba(X_val_gbm)[:,1]
probabilitythreshold_clf = 0.25
y_pred_threshold_clf = (lgbm_clf_pred_proba>=probabilitythreshold_clf).astype(int)
confusion_matrix = pd.crosstab(y_val_gbm, y_pred_threshold_clf, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt = '')

print('Accuracy:',metrics.accuracy_score(y_val_gbm,y_pred_threshold_clf))
print(classification_report(y_val_gbm,y_pred_threshold_clf))


# #### AUC 


fpr, tpr, _ = metrics.roc_curve(y_val_gbm,  lgbm_clf_pred_proba)
auc_gbm = metrics.roc_auc_score(y_val_gbm, lgbm_clf_pred_proba)


plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % auc_gbm)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




feature_imp_lgbm = pd.Series(lgbm_clf.feature_importances_,index=X.columns.values).sort_values(ascending=False)
feature_imp_lgbm

feature_imp_lgbm.plot.bar()


# ### AUC Summary 

print('AUC of Logistic Regression is:', auc_lr)
print('AUC of Classification Tree is:', auc_clf)
print('AUC of Random Forest is:', auc_rf)
print('AUC of LightGBM is:', auc_gbm)


# ## Final model selection
# ###### According to AUC, Random Forest has the best performance. Therefore, the chosen model will be Random Forest. 

# #### Test Set

test_rf = pd.get_dummies(test, columns = ['Region_Code'])
test_rf = test_rf.drop(["id", "Region_Code_50"], axis = 1)
test_rf['Gender']=test_rf['Gender'].replace({'Male':1,'Female':0})
test_rf['Vehicle_Damage'].replace({'Yes':1,'No':0}, inplace=True)
test_rf['Vehicle_Age'].replace({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3}, inplace=True)
test_rf['Age']=np.log(test_rf['Age'])
test_rf['Annual_Premium']=np.log(test_rf['Annual_Premium'])
test_rf['Vintage']=np.log(test_rf['Vintage'])

X_test_rf = test_rf.loc[:, test_rf.columns != 'Response']
y_test_rf = np.ravel(test_rf.loc[:, test_rf.columns == 'Response'])


# ### Fitting test set to Random Forest Model 


y_pred_prob_rf = RandomForestCLF.predict_proba(X_test_rf)[:, 1]
probability_threshold_rf = 0.25
y_pred_threshold_rf = (y_pred_prob_rf >= probability_threshold_rf).astype(int)

confusion_matrix_rf = pd.crosstab(y_test_rf, y_pred_threshold_rf, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix_rf, annot=True, fmt='g')

print('Accuracy: ',metrics.accuracy_score(y_test_rf, y_pred_threshold_rf))
print(classification_report(y_test_rf, y_pred_threshold_rf))


# #### AUC



fpr, tpr, threshold = metrics.roc_curve(y_test_rf, y_pred_prob_rf)
roc_auc = metrics.auc(fpr, tpr)
auc_rf = metrics.roc_auc_score(y_test_rf, y_pred_prob_rf)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Benefit structure 

# ### Benefit stucture 1
# 
# - Promote to an interested customer + 10
# - Miss an interested customer - 10
# - Promote to an uninterested customer - 2
# - Each promotion - 1

# #### Calculate Benefit Structure score 

def benefit_str_1(probability_threshold_1):
    y_pred_prob_rf = RandomForestCLF.predict_proba(X_test_rf)[:, 1]
    probability_threshold_rf = probability_threshold_1
    y_pred_threshold_rf = (y_pred_prob_rf >= probability_threshold_rf).astype(int)
    confusion_matrix_rf = pd.crosstab(y_test_rf, y_pred_threshold_rf, rownames=['Actual'], colnames=['Predicted'])
    if sum(confusion_matrix_rf.loc[:,0]) != 10000: 
        points = (sum(confusion_matrix_rf.loc[:,1])*-1) + (confusion_matrix_rf.loc[0,1]*-2) + (confusion_matrix_rf.loc[1,0]* -10) + (confusion_matrix_rf.loc[1,1]*10)
    else:
        points = confusion_matrix_rf.loc[1,0]*-10
    print( 'Threshold =', probability_threshold_1,': Total Benefit Structure Score is', points)


benefit_str_1(0.01)
benefit_str_1(0.1)
benefit_str_1(0.2)
benefit_str_1(0.5)


# ###  Benefit structure 2
# - Promote to an interested customer + 100
# - Miss an interested customer - 100
# - Promote to an uninterested customer - 2
# - Each promotion - 1

# #### Calculate benefit structure score


def benefit_str_2(probability_threshold_2):
    y_pred_prob_rf = RandomForestCLF.predict_proba(X_test_rf)[:, 1]
    probability_threshold_rf = probability_threshold_2
    y_pred_threshold_rf = (y_pred_prob_rf >= probability_threshold_rf).astype(int)
    confusion_matrix_rf = pd.crosstab(y_test_rf, y_pred_threshold_rf, rownames=['Actual'], colnames=['Predicted'])
    if sum(confusion_matrix_rf.loc[:,0]) != 10000: 
        points = (sum(confusion_matrix_rf.loc[:,1])*-1) + (confusion_matrix_rf.loc[0,1]*-2) + (confusion_matrix_rf.loc[1,0]* -100) + (confusion_matrix_rf.loc[1,1]*100)
    else:
        points = confusion_matrix_rf.loc[1,0]*-100
    
    print( 'Threshold =', probability_threshold_2,': Total Benefit Structure Score is', points)


benefit_str_2(0.01)
benefit_str_2(0.1)
benefit_str_2(0.2)
benefit_str_2(0.5)





