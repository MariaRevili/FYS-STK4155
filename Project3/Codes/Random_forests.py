import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import os


##Prepare the data

url = "https://raw.githubusercontent.com/MariaRevili/FYS-STK4155/master/Project3/eyeData.csv"
data = pd.read_csv(url)
data = data.iloc[:, 1:]
X = data.iloc[:, 0:14]
#print(X.head(3))
y = data.iloc[:, 14]
data = pd.DataFrame(data)
#print(y.head(3))
# print(data["eyeDetection"].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


## Run Random forests 

np.random.seed(20)
rf=RandomForestClassifier(n_estimators=500, bootstrap=False) ##Number of Trees to build
rf.fit(X_train, y_train)
y_rf_predict = rf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_rf_predict))

X_train = pd.DataFrame(X_train)
## Visualize
# tree.plot_tree(rf.estimators_[0], feature_names=X_train.columns, filled=True)
# plt.show()


###Plot the ROC curve
y_pred_prob = rf.predict_proba(X_test)
y_pred_prob = y_pred_prob[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic, Random Forests')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'Plots', 'roc_rf.png'), transparent=True, bbox_inches='tight')
plt.show()



