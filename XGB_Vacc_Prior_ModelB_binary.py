from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import csv
import shap
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
import pickle
import m2cgen as m2c
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import recall_score
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import time


X=pd.read_csv("B/X_2403_b1.csv")
y=pd.read_csv("B/Y_2403_b1.csv")
Xfeat=pd.read_csv("A/X_0803_a.csv")






tot = pd.concat([X,y],axis=1)
tot=tot.sample(frac=1, random_state=1)
tot = tot[tot['annonascita'].notna()]


groups=tot['MMG_x']

y=tot['priorita']

#ycat=tot['priorita_cat']


X=tot.drop(['priorita','MMG_y'],axis=1)



#X=tot[['annonascita']]
anno=X['annonascita'].to_numpy()
anno=anno[groups==1]

X=X.fillna(0)

predictorsdf=pd.DataFrame(X.columns)
predictors = Xfeat.columns




logo = LeaveOneGroupOut()

fold = 0
outer_mae = []
outer_mse = []
outer_r2score = []
ypredtot=[]
gttot=[]
nrowss = 2
ncolss = X.shape[1]
importance_tot = np.zeros((ncolss,1))
importance_Perm = np.zeros((ncolss,1))
importance_Shap = np.zeros((ncolss,1))


X=X.to_numpy()
y=y.to_numpy()
#ycat=ycat.to_numpy()
groups=groups.to_numpy()


yy=y[groups==1]
anno0=anno[yy==1]
anno1=anno[yy==3]
print(mannwhitneyu(anno0, anno1))

ypredtot=[]
gttot=[]
groups_seltot=[]

nrows = len(np.unique(y))
ncols = len(np.unique(y))
cm_tot = np.zeros((nrows, nrows))
outer_recall = []
outer_acc=[]


elapsedalltr = []
elapsedallte = []

for train_index, test_index in logo.split(X, y, groups):
    fold = fold + 1
    print('Fold:')
    print(fold)
    X_train_outer, X_test_outer = X[train_index], X[test_index]
    y_train_outer, y_test_outer = y[train_index], y[test_index]
    #ycat_train_outer, ycat_test_outer = ycat[train_index], ycat[test_index]

    groups_sel = groups[test_index]


  
    countWorsening = np.sum(y_train_outer==20)
    countImprovement = np.sum(y_train_outer==26)
    imbalance_ratio=countWorsening/countImprovement

    
    model = DecisionTreeClassifier(random_state=1)


    # scoring = 'r2'
    t = time.time()
    model.fit(X_train_outer, y_train_outer)
    
    elapsedalltr.append(time.time() - t)
    #best_pars = model.best_params_
    #print(best_pars)

    #best_model = model.best_estimator_
    #best_model.fit(X_train_outer, y_train_outer)
    t = time.time()
    y_pred = model.predict(X_test_outer)
    
    elapsedallte.append(time.time() - t)
    cm = confusion_matrix(y_test_outer, y_pred, labels=[20, 26])
    print(cm)
    cm_tot = cm_tot + cm
    perm_importance = permutation_importance(model, X_test_outer, y_test_outer)
    outer_recall.append(recall_score(y_test_outer, y_pred, average='macro'))
    outer_acc.append(accuracy_score(y_test_outer, y_pred))


    gttot=np.append(gttot,y_test_outer)
    ypredtot=np.append(ypredtot,y_pred)
    groups_seltot=np.append(groups_seltot,groups_sel)

    temp=model.feature_importances_
    importance_tot = importance_tot + temp.reshape(-1, 1)
    importance_Perm=importance_Perm + perm_importance.importances_mean.reshape(-1, 1)




print("Recall")
print((np.mean(outer_recall)))
print((np.std(outer_recall)))

print("Acc")
print((np.mean(outer_acc)))
print((np.std(outer_acc)))

# indices = np.argsort(importance_tot.flatten())[::-1]
# #indices = indices.ravel()
# indices = indices[:10]
# names = [predictors[i] for i in indices]
# importance_tot = np.transpose(importance_tot)
# importance_tot = importance_tot.ravel()
# plt.figure()
# plt.barh(range(indices.shape[0]), importance_tot[indices])
# plt.yticks(range(indices.shape[0]), names)
# plt.xlabel('Importance')
# plt.savefig('Imp_relModelB7.pdf', bbox_inches='tight')

#plt.savefig('REGR_XGB_feat_importance_Vacc2.png', bbox_inches='tight')
# plt.tight_layout()
# plt.show()
##
print(cm_tot)
print(accuracy_score(gttot, ypredtot))
print('Recall: %.4f' % (np.mean(outer_recall)))


print(np.mean(elapsedalltr))
print(np.std(elapsedalltr))


print(np.mean(elapsedallte))
print(np.std(elapsedallte))
    
# indices = np.argsort(importance_Perm.flatten())[::-1]
# #indices = indices.ravel()
# indices = indices[:10]
# names = [predictors[i] for i in indices]
# importance_tot = np.transpose(importance_Perm)
# importance_tot = importance_Perm.ravel()
# plt.figure()
# plt.barh(range(indices.shape[0]), importance_tot[indices])
# plt.yticks(range(indices.shape[0]), names)
# plt.xlabel('Imp_relModelPermB2')

# save results to csv
# with open('REGR_XGB_results_Neuro.csv', 'w') as csvfile:
#     fieldnames = ['mae', 'mse', 'r2score']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerow({'mae': np.mean(outer_mae, dtype=np.float16), 'mse': np.mean(outer_mse, dtype=np.float16),
#                      'r2score': np.mean(outer_r2score, dtype=np.float16)})
# ##


# plt.figure()
# y_pred=y_pred.reshape(-1, 1)
# plt.plot(y_test_outer[0:200], label = "Ground-truth",marker='o', markersize=3, color='red', linewidth=0)
# plt.plot(y_pred[0:200], label = "Predicted",color='blue', linewidth=1)
# plt.xlabel('Patients')
# # Set the y axis label of the current axis.
# plt.ylabel('Priority')
# plt.savefig('Trend_rel2.png', bbox_inches='tight')


# countWorsening = np.sum(y==2)
# countImprovement = np.sum(y==1)
# imbalance_ratio = countImprovement / countWorsening

# model=xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=1, learning_rate=0.1,
#                           max_depth=5, alpha=0, n_estimators=10, random_state=1, nthread=-1, scale_pos_weight=imbalance_ratio)

# model.fit(X, y)

# pickle.dump(model, open("model0303/modelB/model_XGB_modelB1.model", "wb"))
# model.save_model("model0303/modelB/model_XGB_modelB1.json")

# model_exportedphp = m2c.export_to_php(model)

# #model_XGB1_Neuro_c_sharp = m2c.export_to_c_sharp(best_model)
# with open("model0303/modelB/model_XGB_modelB1.txt", "w") as output:
#     output.write(model_exportedphp)
    

#ax2.plot('Ground-truth', data=datapreproc, marker='o', markersize=12, color='red', linewidth=4)
#ax2.plot( 'Predicted', data=datapreproc, marker='*', markersize=12, color='blue', linewidth=2)
#ax2.set_xlabel('observations (tl, pt, frindex)')
#ax2.set_ylabel('Priority')