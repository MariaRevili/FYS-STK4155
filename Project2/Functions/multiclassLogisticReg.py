import numpy as np

class multiclassLogistic:
    
    def __init__(self, X, y, y_onehot, learning_rate, lambda_):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.y_onehot = y_onehot
        
        
    def softmax(self, X) :
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    
    def sgd(self, X, y, y_onehot, iterations, lambda_, learning_rate):
        w = np.zeros([X.shape[1],len(np.unique(y))])        
        losses = []
        
        for i in range(0,iterations):
            loss,grad = self._getLoss(w,X,y_onehot, lambda_)
            losses.append(loss)
            w = w - (learning_rate * grad)
        return w
    
    
    def _getLoss(self, w, X, y_onehot, lambda_):
        m = X.shape[0] #First we get the number of training examples
        scores = np.dot(X,w) #Then we compute raw class scores given our input and current weights
        prob = self.softmax(scores) #Next we perform a softmax on these scores to get their probabilities
        loss = (-1 / m) * np.sum(y_onehot * np.log(prob)) + (lambda_/2)*np.sum(w*w) #We then find the loss of the probabilities
        grad = (-1 / m) * np.dot(X.T,(y_onehot - prob)) + lambda_*w #And compute the gradient for that loss
        return loss,grad
    
    
    def accuracy(self, someX,someY, w):
        prob,prede = self._getProbsAndPreds(someX, w)
        accuracy = sum(prede == someY)/(float(len(someY)))
        return accuracy
    
    
    def _getProbsAndPreds(self, someX, w):
        probs = self.softmax(np.dot(someX,w))
        preds = np.argmax(probs,axis=1) ##returns the highest value index (i.e. class) along the row
        return probs,preds