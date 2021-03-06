import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent
    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    X = np.array(X)
    y = np.array(y)
    y[y == 0] = -1  # Convert all 0 class labels to -1 to use formula taught in lecture
    X_with_X0 = np.insert(X, 0, 1, axis=1)
    w_with_w0 = np.insert(w, 0, 0)

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0

        average_learning_rate = step_size / N

        for i in range(max_iterations):
            # X -> (350, 3)
            # y -> (350, 1)
            # w -> (3,) numpy array

            predictions = np.dot(X_with_X0,
                                 w_with_w0)  # wTX   -> here it doesn't matter if we do X.w or X.wT because w is of shape (D,) and wT is also same
            indicators = np.multiply(y, predictions)  # yn * wTX
            indicators = np.where(indicators <= 0, 1,
                                  0)  # (350, 1)  -> Transform all misclassified labels: Indicator function
            # gradient_w = sum (indicator <= 0 * y * X)
            product = np.multiply(indicators, y)  # Indicator * y
            gradient_w = np.dot(product, X_with_X0)  # Indicator * y * X
            w_with_w0 += average_learning_rate * gradient_w  # w <- w + gradient_w

        w = w_with_w0[1:]
        b = w_with_w0[0]
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        average_learning_rate = step_size / N

        for i in range(max_iterations):
            predictions = np.dot(X_with_X0, w_with_w0)
            y_preds = np.multiply(y, predictions)
            z = sigmoid(-1 * y_preds)
            z_y = np.multiply(z, y)
            gradient_w = np.dot(z_y, X_with_X0)
            w_with_w0 += average_learning_rate * gradient_w

        w = w_with_w0[1:]
        b = w_with_w0[0]

    else:
        raise Exception("Loss Function is undefined.")

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    return 1 / (1 + np.exp(-z))
    ############################################


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.dot(X, w) + b
        preds = np.where(preds > 0, 1, 0)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.dot(X, w) + b
        preds = sigmoid(preds)
        preds = np.where(preds > 0.5, 1, 0)
        ############################################

    else:
        raise Exception("Loss Function is undefined.")

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent
    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros((C, 1))
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        for i in range(max_iterations):
            random_num = np.random.choice(N)
            y_one_hot = np.zeros((C, 1))
            y_n = y[random_num]
            y_one_hot[y_n] = 1    # One hot encoding

            selected_x_n = X[random_num]    # numpy array -> row of all xi shape (D,)
            dim = len(selected_x_n)

            updated_weights = np.dot(selected_x_n, np.transpose(w)) + np.transpose(b)  # (1, C) -> wT
            probabilities = softmax(np.transpose(updated_weights))     # (C, 1)
            sm_probabilities = probabilities - y_one_hot   # softmax probs (C, 1)
            x_n = selected_x_n.reshape(1, dim)
            delta_w = np.dot(sm_probabilities, x_n)

            w -= step_size * delta_w
            b -= step_size * sm_probabilities

        b = b.reshape(len(w), )


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        # X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
        # w = np.hstack((w, np.ones((w.shape[0], 1), dtype=w.dtype)))

        X_new = np.insert(X, np.shape(X)[1], 1, axis=1)
        w_new = np.insert(w, np.shape(w)[1], b, axis=1)

        y_one_hot_matrix = np.eye(C)[y]
        avg_learn_rate = step_size / N
        for i in range(max_iterations):
            updated_weights = np.matmul(w_new, np.transpose(X_new))
            soft_max = softmax(updated_weights) - np.transpose(y_one_hot_matrix)
            w_new -= avg_learn_rate * np.matmul(soft_max, X_new)

        b = w_new[:, -1]
        w = np.delete(w_new, -1, axis=1)

    else:
        raise Exception("Type of Gradient Descent is undefined.")

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    X_new = np.insert(X, np.shape(X)[1], 1, axis=1)
    w_new = np.insert(w, np.shape(w)[1], b, axis=1)
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################
    preds = np.matmul(w_new, np.transpose(X_new))
    preds = np.argmax(preds, axis=0)
    assert preds.shape == (N,)
    return preds


def softmax(x):
    x -= np.max(x, axis=0)
    soft_max = np.exp(x) / np.sum(np.exp(x), axis=0)
    return soft_max