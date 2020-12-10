import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 
import os




###Download the data
url = "https://raw.githubusercontent.com/MariaRevili/FYS-STK4155/master/Project3/eyeData.csv"
data = pd.read_csv(url)
data = data.iloc[:, 1:]
X = data.iloc[:, 0:14]
#print(X.head(3))
y = data.iloc[:, 14]
data = pd.DataFrame(data)
#print(y.head(3))
# print(data["eyeDetection"].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

##scale the data 

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


### Obtain predictions from each weak tree and then use these predictions as covariates for neural networks
np.random.seed(20)
pred_train = np.ones((y_train.size, 1))
pred_test = np.ones((y_test.size, 1))


rf=RandomForestClassifier(n_estimators=500, bootstrap=False) ##Number of Trees to build
rf.fit(X_train, y_train)

for tr in rf.estimators_:
    per_tree_pred_tr = tr.predict(X_train).reshape(-1,1)
    per_tree_pred_te = tr.predict(X_test).reshape(-1,1)
    pred_train = np.c_[pred_train, per_tree_pred_tr]
    pred_test = np.c_[pred_test, per_tree_pred_te]


pred_train = pd.DataFrame(pred_train)
pred_train = pred_train.iloc[:, 1:]

pred_test = pd.DataFrame(pred_test)
pred_test = pred_test.iloc[:, 1:]



###Now learn these trees with boosting
max_depth = 3
ab=AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth), n_estimators=500) ##Number of Trees to build
ab.fit(pred_train, y_train)
y_ab_predict = ab.predict(pred_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_ab_predict))

pred_train_ = np.ones((y_train.size, 1))
pred_test_ = np.ones((y_test.size, 1))


##predict from each tree
for tree in ab.estimators_:
    per_tree_pred_tr = tree.predict(pred_train).reshape(-1,1)
    per_tree_pred_te = tree.predict(pred_test).reshape(-1,1)
    pred_train_ = np.c_[pred_train_, per_tree_pred_tr]
    pred_test_ = np.c_[pred_test_, per_tree_pred_te]


pred_train_ = pd.DataFrame(pred_train)
pred_train_ = pred_train.iloc[:, 1:]

pred_test_ = pd.DataFrame(pred_test)
pred_test_ = pred_test.iloc[:, 1:]






####Feed these predicted values (on the test data) instead of 
# original covariates into neural networks, estimate accuracy on the train data

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental import preprocessing


epochs = 500
batch_size = 100
n_neurons_layer1 = 100
eta_vals = np.logspace(-5, 0, 5)
lmbd_vals = np.logspace(-5, 0, 5)

def neural_network_keras(n_neurons_layer1, eta, lmbd):  ##Build the model
    
    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation='relu', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])  ##try adam also
    
    return model

DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
  

##Tune the parameters
def train_dnn():    ##fit for different learning rate and decay (lambda) 
     
    for i, eta in enumerate(eta_vals):
        
        for j, lmbd in enumerate(lmbd_vals):
            
            DNN = neural_network_keras(n_neurons_layer1, 
                                            eta=eta, lmbd=lmbd)
            DNN.fit(pred_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0) 
            y_pred = DNN.predict_classes(pred_test)
            scores = DNN.evaluate(pred_test, y_test, verbose=0)
            
            DNN_keras[i][j] = DNN
            
            print("Learning rate = ", eta)
            print("Lambda = ", lmbd)
            print("Test MSE = ", scores)
            print()  
  
#train_dnn()  


## Show the accuracy after training

DNN = neural_network_keras(n_neurons_layer1, 
                           eta=1, lmbd=0.003)
DNN.fit(pred_train_, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0) 
y_pred = DNN.predict_classes(pred_test_)
scores = DNN.evaluate(pred_test_, y_test, verbose=1)
print("Test Accuracy NNets = ", scores)


## Plot the ROC curve

# y_pred_prob = DNN.predict_proba(pred_test)
# fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob)
# roc_auc = metrics.auc(fpr, tpr)

# plt.title('Receiver Operating Characteristic, Deep Forest Learning')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
#plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'Plots', 'roc_dfl.png'), transparent=True, bbox_inches='tight')
#plt.show()