import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import pickle

LOG_DIR = "logs/sign_nn_with_summaries"

def variable_summaries(var):
    with tf.name_scope('summaries'):
     mean = tf.reduce_mean(var)
     tf.summary.scalar('mean', mean)

     with tf.name_scope('stddev'):
         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
         tf.summary.scalar('stddev', stddev)
         tf.summary.scalar('max', tf.reduce_max(var))
         tf.summary.scalar('min', tf.reduce_min(var))
         tf.summary.histogram('histogram', var)


# Loading the datasets for SIGN language  where
# train_set_x_orig is array of  1080 image samples where each image is 64X64X3=12288 ==> dimension : [12288,1080]
# train_set_y_orig is array of labels for each of 1080 image. ==> dimension: [6,1080], 6 classes are from SIGN
# Image number from 0-5
#  Also here for test sets,
# test_set_x_orig ==> array of 120 images of SIGN ==> Dimension : [12288,120]
# test_set_y_orig ===>array of 120 labels for each of above image : [6,120]

# here we would implement following model
# LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# selecting rows  of np.eye(6) mapped to columns values in Y.reshape(-1).T ,
# total 1080 values for train and 120 for test
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3

# Loading the dataset
def get_data():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    return X_train, X_test, Y_train, Y_test

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
         Creates a list of random minibatches from (X, Y)

         Arguments:
         X -- input data, of shape (input size, number of examples)
         Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
         mini_batch_size - size of the mini-batches, integer
         seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

         Returns:
         mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Randomly permute a sequence of m (m=1080 samples for training and m=120 samples  for test images )
    permutation = list(np.random.permutation(m))  # generate permuted sequence of integeres upto m
    shuffled_X = X[:, permutation]  # Step 1: Shuffle (X, Y)
    # shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)

    # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


def forward_propagation_for_predict(X, parameters):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32,shape=[n_x,None])
    Y = tf.placeholder(tf.float32,shape=[n_y,None])
    return X,Y

# last layer would have 6 nodes as this is multi-class classification problem .
def initialize_parameters(n1,n2):
    tf.set_random_seed(1)
    with tf.name_scope("W_1"):
        W1 = tf.get_variable("W1", [n1, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))  # 25 hidden nodes
        variable_summaries(W1)
    with tf.name_scope("b_1"):
        b1 = tf.get_variable("b1", [n1, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        variable_summaries(b1)
    with tf.name_scope("W_2"):
        W2 = tf.get_variable("W2", [n2, n1], initializer=tf.contrib.layers.xavier_initializer(seed=1))  # 12 hidden nodes
        variable_summaries(W2)
    with tf.name_scope("b_2"):
        b2 = tf.get_variable("b2", [n2, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        variable_summaries(b2)
    with tf.name_scope("W_3"):
        W3 = tf.get_variable("W3", [6, n2], initializer=tf.contrib.layers.xavier_initializer(seed=1))  # 6 hidden nodes
        variable_summaries(W3)
    with tf.name_scope("b_3"):
        b3 = tf.get_variable("b3", [6, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        variable_summaries(b3)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1) # Dimension : [25,12288] * [12288,1080] = [25,1080]
    A1 = tf.nn.relu(Z1) # [25,1080]
    Z2 = tf.add(tf.matmul(W2, A1), b2) # Dimension :  [12,25] * [25,1080] = [12,1080]
    A2 = tf.nn.relu(Z2) # [12,1080]
    Z3 = tf.add(tf.matmul(W3, A2), b3) # Dimension : [6,12] * [12,1080] = [6,1080]

    return Z3


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3) # Dimension :  [1080,6]
    labels = tf.transpose(Y) # Dimension :  [1080,6]
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    print("logits shape=" + str(logits.shape))
    print("labels shape=" + str(labels.shape))
    return cost


def model(n1,n2,X_train, Y_train, X_test, Y_test, learning_rate=0.0001,num_epochs=1000, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()

    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n1,n2)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    merged = tf.summary.merge_all()
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                             graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                                            graph=tf.get_default_graph())
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            i = 0;
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                summary,_, minibatch_cost = sess.run([merged,optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
                with tf.name_scope("COST"):
                    variable_summaries(epoch_cost)
                i = i + 1;
                train_writer.add_summary(summary, i)
                test_writer.add_summary(summary,i)

            if print_cost == True :
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:
                 costs.append(epoch_cost)

        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

def predictImage(image,parameters):
    im = Image.open(image)
    imagearr = np.array(im) #create an np array of image
    imagearr.resize(64,64,3) #changes value of imagearr
    my_image = imagearr.reshape(1,64 * 64 * 3).T #changes to (12288,1) size
    my_image_prediction = predict(my_image, parameters)

    print("prediction of image is " + str(my_image_prediction))

    #im.show()

def train_params(n1,n2,need_train):
   if(need_train == True):
        parameters = model(n1,n2,X_train, Y_train, X_test, Y_test)
        with open("models/sign_nn/params.pickle", "wb") as f :
            pickle.dump(parameters, f)

   else:
        with open("models/sign_nn/params.pickle", "rb") as f:
                parameters = pickle.load(f)

   return parameters

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = get_data()
    #n1 = number of hidden nodes in first layer
    #n2 = number of hidden nodes in second layer
    parameters = train_params(n1=25,n2=12,need_train=True)
    predictImage("thumbs_up.jpg",parameters)

