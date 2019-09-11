import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    if predictions.ndim == 2:
        pred = predictions.copy()
        pred -= pred.max(axis=1)[:,None]
        sum = np.sum(np.exp(pred), axis = 1)
        result = np.exp(pred)/sum[:,None]
        return result
    elif predictions.ndim == 1:
        pred = predictions.copy()
        pred -= pred.max()
        sum = np.sum(np.exp(pred))
        result = np.exp(pred)/sum
        return result


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if target_index.ndim == 2:

        m = probs.shape[0]
        p = probs
        log_likelihood = np.zeros(m)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        for i in range(probs.shape[0]):
            log_likelihood[i] = -np.log(p[i,target_index[i]])
        loss = np.sum(log_likelihood) / m

    elif target_index.ndim == 1:
        loss = -np.log(probs[target_index]) / target_index.shape

    result = loss
    return result

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = np.zeros(predictions.shape)

    if predictions.ndim == 2:
        m = predictions.shape[0]
        grad = probs.copy()
        for i in range(m):
            grad[i, target_index[i]] -= 1

        grad = grad / m
        return loss, grad

    elif predictions.ndim == 1:
        for i in range(probs.shape[0]):
            if target_index == i:
                dprediction[i] = probs[i] - 1
            else:
                dprediction[i] = probs[i]

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops

    loss = reg_strength * np.sum(np.square(W))

    grad = W.copy()
    grad = 2*reg_strength*grad

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes probs
    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''

    predictions = np.dot(X, W)
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)



    m = target_index.shape[0]
    for i in range(m):
        probs[i, target_index[i]] -= 1

    grad = np.zeros(W.shape)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            #grad[i,j] = probs[i,] * X[j,i]
            m = probs.shape[0]
            grad[i, j] = np.sum (X[range(m),i] * probs[range(m), j])

    grad = grad / m

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    print ("loss = ", loss)
    print ("grad = ", grad)

    return loss, grad


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            loss, gradient = linear_softmax(X,self.W,y)
            l2_reg, l2_reg_grad = l2_regularization(self.W, reg)
            self.W = self.W - learning_rate * gradient + l2_reg_grad
            loss_history.append(loss)

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops

        res = np.dot(X,self.W)

        for i in range(res.shape[0]):
            y_pred[i] = np.argmax(res[i,:])


        return y_pred