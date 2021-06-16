
# MonteCarlo_simulation_average_treatment_effects_using_NNs.py
# -----------------------------------------------------------------
# Date: November 2018
# Last modified: July 2020
# This is the Python code for a Monte Carlo simulation used to support
# findings of Farrell, M.H., Liang, T. and Misra, S., 2019:
# 'Deep neural networks for estimation and inference',
# arXiv preprint arXiv:1809.09953.
# ------------------------------------------------------------------


import os
# Stopping Tensorflow from printing info messages and warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import combinations_with_replacement
import tensorflow as tf
import logging
import time
import random
import sys

# Stopping deprecation warnings
logging.getLogger('tensorflow').disabled = True

'''
FLAGS options
-------------
    FLAGS.update: bool
        If True the simulation results will be saved in appropriate
        .csv file.
    FLAGS.plot_true: bool
        Turn plotting on or off. If many simulations are run, set
        FLAGS.plot_true = False since the plots are saved in memory
        until the program finishes, which can exhaust memory.
    FLAGS.verbose: bool
        Turn printing detailed messages on or off. If you run more than
        a few simulations, you probably want to turn it off.
    FLAGS.nsimulations: int
        Number of simulations to run.
    FLAGS.nconsumer_characteristics: int
        Number of consumer characteristics in the artificial dataset.
        In our simulations, we used 20 and 100 consumer characteristics.
        It shouldn't be less than 20.
    FLAGS.treatment: {'random' , 'not_random'}
        If 'random', consumers are being treated at random. Otherwise,
        probability of being treated is a function of consumer
        characteristics.
    FLAGS.model: {'simple' , 'quadratic'}
        If 'simple', coefficients a and b in the artificial dataset
        depend linearly on consumer characteristics. Otherwise, the
        dependence is quadratic.
    FLAGS.architecture: {architecture_1_, architecture_2_, ... ,
        architecture_9_}
        Which NN architecture to use. See below for parameters of these
        architectures. To experiment with other NN architectures either
        edit one of the existing architecture parameters or add
        'architecture_10_' and define parameters in a similar manner.
        Make sure to change architecture flag parameter value below as
        well. Options for activations functions are 'relu', 'lrelu',
        'prelu','srelu', 'plu', 'elu', 'none'.
    FLAGS.data_seed: int or None
        Which seed number to use for creation of fake dataset. If it is
        set to None, a random seed is used.

Model parameters
----------------
    train_proportion: scalar
        A proportion of the dataset to be used for training. Has to be
        between 0 and 1. If it is set to 1, than early_stopping is set
        to False and the NN will be trained on the whole dataset for the
        max_nepochs. The parameters will be retrieved at the last epoch.
        Otherwise, training will be stopped when there is no improvement
        of the loss on the validation set for max_epochs_without_change
        or when max_nepochs is reached. The parameters will be retrieved
        at the epoch where best validation loss is recorded.
    max_nepochs: int
        Maximum number of epochs for which NNs will be trained.
    max_epochs_without_change: int
        Number of epochs with no improvement on the validation loss to
        wait before stopping the training.
    hidden_layer_sizes: list of ints
        Hidden layers for the first NN that estimates the treatment
        coefficients. Length of the list defines the number of hidden
        layers. Entries of the list define the size of each hidden
        layer.
    activation_functions: list of {'relu', 'lrelu', 'prelu','srelu',
        'plu', 'elu', 'none'}
        Which activation function to use on each hidden layer of the
        first NN. Has to be of length len(hidden_layer_sizes) + 1. The
        last element in the list should be 'none', because it
        corresponds to the output layer.
    dropout_rates_train: list of floats
        Dropout rate on input and each hidden layer of the first NN.
        Has to be of length len(hidden_layer_sizes) + 1 .
    hidden_layer_sizes_treatment: list of ints
        Hidden layers for the second NN that estimates the propensity
        scores.
    activation_functions_treatment: list of {'relu', 'lrelu', 'prelu',
        'srelu', 'plu', 'elu', 'none'}
        Which activation function to use on each hidden layer of the
        second NN.
    dropout_rates_train_treatment: list of floats
        Dropout rate on each hidden layer of the second NN.
    optimizer: {'RMSProp' , 'GradientDescent' , 'Adam'}
        Which optimizer to use. In all of the simulations reported in
        the paper, the 'Adam' optimizer was used.
    learning_rate: scalar
        Learning rate.
    batch_size: int or None
        Batch size. If int, batch_size should be smaller than the
        length of the training set. To train on the whole training set
        rather than on mini-batches, set to None.
    alpha: float
        Regularization strength parameter.
    r: float
        Mixing ratio of Ridge and Lasso regression. If it's equal to 0.,
        than the regularization is equal to Ridge regression. If it is
        equal to 1., it is equal to Lasso regression.
    nconsumers: int
        Number of consumers.
'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('update', False,
                            """Record the simulation results.""")
tf.app.flags.DEFINE_boolean('plot_true', False, """Show plots.""")
tf.app.flags.DEFINE_boolean('verbose', True, """Show detailed messages.""")
tf.app.flags.DEFINE_integer('nsimulations', 1,
                            """How many simulations to run.""")
tf.app.flags.DEFINE_integer('nconsumer_characteristics', 100,
                            """Number of consumer characteristics.""")
tf.app.flags.DEFINE_string('treatment', 'not_random',
                           """Are customers treated at random or not.""")
tf.app.flags.DEFINE_string('model', 'quadratic',
                           """Is the mapping from consumer characteristics
                           to their preferences linear or quadratic.""")
tf.app.flags.DEFINE_string('architecture', 'architecture_1_',
                           """Which NN architecture to use.""")
tf.app.flags.DEFINE_integer('data_seed', None,
                            """Seed to use to create fake data.""")

remaining_args = FLAGS([sys.argv[0]] +
                       [flag for flag in sys.argv if flag.startswith("--")])
assert (remaining_args == [sys.argv[0]])

# Different architectures for the first NN
if FLAGS.architecture == 'architecture_1_':
    hidden_layer_sizes = [20, 10, 5]
    dropout_rates_train = [0, 0, 0, 0]
    activation_functions = ['relu', 'relu', 'relu', 'none']

elif FLAGS.architecture == 'architecture_2_':
    hidden_layer_sizes = [60, 30, 20]
    dropout_rates_train = [0, 0, 0, 0]
    activation_functions = ['relu', 'relu', 'relu', 'none']

elif FLAGS.architecture == 'architecture_3_':
    hidden_layer_sizes = [80, 80, 80]
    dropout_rates_train = [0, 0, 0, 0]
    activation_functions = ['relu', 'relu', 'relu', 'none']

elif FLAGS.architecture == 'architecture_4_':
    hidden_layer_sizes = [20, 15, 10, 5]
    activation_functions = ['relu', 'relu', 'relu', 'relu', 'none']
    dropout_rates_train = [0, 0, 0, 0, 0]

elif FLAGS.architecture == 'architecture_5_':
    hidden_layer_sizes = [60, 30, 20, 10]
    activation_functions = ['relu', 'relu', 'relu', 'relu', 'none']
    dropout_rates_train = [0, 0, 0, 0, 0]

elif FLAGS.architecture == 'architecture_6_':
    hidden_layer_sizes = [80, 80, 80, 80]
    activation_functions = ['relu', 'relu', 'relu', 'relu', 'none']
    dropout_rates_train = [0, 0, 0, 0, 0]

elif FLAGS.architecture == 'architecture_7_':
    hidden_layer_sizes = [20, 15, 15, 10, 10, 5]
    dropout_rates_train = [0, 0, 0, 0, 0, 0, 0]
    activation_functions = [
        'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'none'
    ]

elif FLAGS.architecture == 'architecture_8_':
    hidden_layer_sizes = [60, 30, 20, 20, 10, 5]
    dropout_rates_train = [0, 0, 0, 0, 0, 0, 0]
    activation_functions = [
        'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'none'
    ]

elif FLAGS.architecture == 'architecture_9_':
    hidden_layer_sizes = [80, 80, 80, 80, 80, 80]
    dropout_rates_train = [0, 0, 0, 0, 0, 0, 0]
    activation_functions = [
        'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'none'
    ]
else:
    raise ValueError('Architecture not found! Check the spelling.')

if FLAGS.nconsumer_characteristics < 20:
    raise ValueError('Number of consumer characteristics ' +
                     'should not be less than 20.')

dropout_rates_test = [0 for i in dropout_rates_train]

# Architecture for the second NN that estimates
# propensity scores
hidden_layer_sizes_treatment = [50, 30]
activation_functions_treatment = ['relu', 'relu', 'none']
dropout_rates_train_treatment = [0, 0, 0]
dropout_rates_test_treatment = [0 for i in dropout_rates_train_treatment]

# Setting parameters values for generating fake data
nconsumers = 10000

# Run parameters
train_proportion = 0.9
max_nepochs = 5000
max_epochs_without_change = 30

if train_proportion == 1:
    early_stopping = False
else:
    early_stopping = True

optimizer = 'Adam'
learning_rate = 0.009
batch_size = 128
batch_size_t = None

# Regularization parameters
alpha = 0.
r = 0.2

# Checking for spelling errors
if not (FLAGS.model == 'quadratic' or FLAGS.model == 'simple'):
    raise ValueError('Check whether model type is spelled correctly!')
if not (FLAGS.treatment == 'random' or FLAGS.treatment == 'not_random'):
    raise ValueError('Check whether treatment type is spelled correctly!')
    
start_time = time.time()

X_train = T_train = Y_train = X_valid = \
    T_valid = Y_valid = X = T_real = Y = None


# ---------------------- Function definitions ------------------------
def running_time():
    '''
    Print the time passed since start_time.
    '''
    end_time = time.time()
    hours = int((end_time-start_time) / 3600)
    minutes = int((end_time-start_time) % 3600 / 60)
    seconds = int(end_time - start_time - (3600*hours + 60*minutes))
    print('Running time is: {} hours, {} minutes and {} seconds'.format(
        hours, minutes, seconds))


class FakeData():
    '''
    Create an artificial dataset with desired properties for testing
    the NN method.

    Inputs:
    -------
        N: int
            Number of consumers.
        nconsumer_characteristics: int
            Number of consumer characteristics.
        data seed: int or None
            Seed used to create fake dataset. If it is set to None
            than random seed is used.
    '''
    def __init__(self, N=nconsumers,
                 nconsumer_characteristics=FLAGS.nconsumer_characteristics,
                 data_seed=FLAGS.data_seed):

        self.N = N
        self.nconsumer_characteristics = nconsumer_characteristics
        self.seed = data_seed

        # Creating fake data variables
        self.Y = None
        self.X = None
        self.mu0 = None
        self.tau = None
        self.T = None
        self.prob_of_T = None
        self.tau_true_mean = None

    def _sum_polynomial_X_times_weights(self, weights):
        '''
        Evaluate the non-linear part of a quadratic polynomial in
        consumer characteristics, self.X, with prescribed weights, w.

        Inputs:
        -------
            weights: ndarray, shape = (num_additional_poly_terms, )
                Weights corresponding to quadratic terms.
        Outputs:
        -------
            sum_x: ndarray, shape = (N, 1)
                Non-linear part of the quadratic polynomial evaluated
                for each consumer.
        '''
        my_polynomial_indices = combinations_with_replacement(
            list(range(self.nconsumer_characteristics)), 2)
        i = 0
        sum_x = 0
        for p in my_polynomial_indices:
            sum_x = sum_x + weights[i]*np.multiply(self.X[:, p[0]],
                                                   self.X[:, p[1]])
            i += 1
        sum_x = sum_x.reshape(-1, 1)
        return sum_x

    def _create_TE_coefs(self, model):
        '''
        Create treatment effect coefficients.

        Inputs:
        -------
            model: {'simple', 'quadratic'}
                If 'simple' coefficients a and b in the artificial
                dataset depend linearly on consumer characteristics.
                Otherwise, the dependence is quadratic.
        Outputs:
        -------
            bias_tau: float
                Constant term in equation for tau.
            alpha_tau: ndarray, shape = [nconsumer_characteristics, 1]
                Linear coefficients in equation for tau.
            beta_tau: ndarray, shape = [count]
                Quadratic coefficients in equation for tau.
                Count is the number of the second degree terms in a
                quadratic polynomial where the number of variables is
                equal to the number of consumer characteristics.
        '''
        np.random.seed(63)

        # Calculating tau
        alpha_tau = np.random.uniform(low=0.1, high=0.22,
                                      size=[self.nconsumer_characteristics, 1])
        bias_tau = -0.05
        self.tau = np.dot(self.X, alpha_tau) + bias_tau

        if model == 'quadratic':
            count = comb(self.nconsumer_characteristics, 2, True, True)
            beta_tau = np.random.uniform(low=-0.05, high=0.06, size=count)
            self.tau = self.tau + self._sum_polynomial_X_times_weights(
                beta_tau)
        else:
            beta_tau = None

        # Calculating mu0
        alpha_mu0 = np.random.normal(loc=0.3, scale=0.7,
                                     size=[1, self.nconsumer_characteristics])
        bias_mu0 = 0.09
        self.mu0 = np.dot(self.X, alpha_mu0.T) + bias_mu0

        if model == 'quadratic':
            beta_mu0 = np.random.normal(loc=0.01,
                                        scale=0.3,
                                        size=count)
            self.mu0 = self.mu0 + self._sum_polynomial_X_times_weights(
                beta_mu0)
        return alpha_tau, bias_tau, beta_tau

    def _calculate_true_tau_mean(self, alpha_tau, bias_tau, beta_tau, model):
        '''
        Calculate true average treatment effect.

        Inputs:
        -------
            bias_tau: float
                Constant term in equation for tau.
            alpha_tau: ndarray, shape = [nconsumer_characteristics, 1]
                Linear coefficients in equation for tau.
            beta_tau: ndarray, shape = [count]
                Quadratic coefficients in equation for tau.
                Count is the number of the second degree terms in a
                quadratic polynomial where the number of variables is
                equal to the number of consumer characteristics.
            model: {'simple', 'quadratic'}
                If 'simple' coefficients a and b in the artificial
                dataset depend linearly on consumer characteristics.
                Otherwise, the dependence is quadratic.
        '''
        X = 0.5

        self.tau_true_mean = np.sum(X * alpha_tau) + bias_tau

        if model == 'quadratic':
            X_poly = 0.25 * np.ones(len(beta_tau))
            s = 0
            for i in range(self.nconsumer_characteristics):
                X_poly[s] = 1/3.
                s = s + self.nconsumer_characteristics - i

            self.tau_true_mean = self.tau_true_mean + np.sum(X_poly * beta_tau)

    def _create_propensity_scores(self, treatment):
        '''
        Calculate propensity scores and create treatment variable for
        our fake dataset.

        Inputs:
        -------
            treatment: {'random', 'not_random'}
                    If 'random' consumers are being treated at random.
                    Otherwise, probability of being treated is a function
                    of consumer characteristics.
        '''
        if treatment == 'random':
            self.prob_of_T = 0.5
            self.T = np.random.binomial(
                size=self.N, n=1, p=self.prob_of_T).reshape(self.N, 1)
        else:
            bias_p = 0.09
            np.random.seed(72)
            alpha_p = np.random.uniform(low=-0.55, high=0.55, size=[20, 1])
            # Probability of t only depends on the first 20 consumers
            # features
            p_of_t = np.dot(self.X[:, :20], alpha_p) + bias_p
            p_of_t = p_of_t.reshape(-1)
            self.prob_of_T = 1 / (1+np.exp(-p_of_t))
            self.T = np.random.binomial(size=self.N, n=1,
                                        p=self.prob_of_T).reshape(self.N, 1)

    def create_fake_data(self, model=FLAGS.model, verbose=FLAGS.verbose,
                         treatment=FLAGS.treatment):
        '''
        Create an artificial dataset.

        Consumer characteristics, X, and normal errors are generated for
        each consumer randomly. Then coefficients mu0 and tau are
        created as functions of consumer characteristics, X.

        If treatment == 'random' all the consumers are treated with an
        equal likelihood of 0.5. Otherwise, the propensity scores that
        depend on consumer characteristics are calculated.

        Finally, a target variable is created.

        Inputs:
        -------
            model: {'simple', 'quadratic'}
                If 'simple' coefficients a and b in the artificial
                dataset depend linearly on consumer characteristics.
                Otherwise, the dependence is quadratic.
            verbose: bool
                If True print detailed messages.
            treatment: {'random', 'not_random'}
                If 'random' consumers are being treated at random.
                Otherwise, probability of being treated is a function
                of consumer characteristics.
        Outputs:
        -------
            self.Y: ndarray, shape = (N, 1)
                Target value.
            self.X: ndarray, shape = (N, nconsumer_characteristics)
                Consumer characteristics.
            self.mu0: ndarray, shape = (N, 1)
                Mu0 for each consumer.
            self.tau: ndarray, shape = (N, 1)
                Tau for each consumer.
            self.T: ndarray, shape = (N, 1)
                Treatment for each consumer.
            self.seed: int
                Random seed used to create fake dataset.
            self.prob_of_T: ndarray, shape = (N,)  or 0.5
                Propensity scores for each consumer if treatment
                variable is set to 'not_random'.
            self.tau_true_mean: float
                True average treatment effect.
        '''
        if self.seed is None:
            self.seed = random.randint(1, 100000)
        if verbose:
            print('Seed number is: ', self.seed)
        np.random.seed(self.seed)

        self.X = np.random.uniform(
            low=0, high=1, size=[self.N, self.nconsumer_characteristics])
        normal_errors = np.random.normal(size=[self.N, 1], loc=0.0, scale=1.0)
        alpha_tau, bias_tau, beta_tau = self._create_TE_coefs(model)
        self._calculate_true_tau_mean(alpha_tau, bias_tau, beta_tau, model)
        self._create_propensity_scores(treatment)
        self.Y = self.mu0 + self.tau*self.T + normal_errors
        return (self.Y, self.X, self.mu0, self.tau, self.T, self.seed,
                self.prob_of_T, self.tau_true_mean)


def get_train_test_inds(t):
    '''
    Split the dataset into training and validation sets while
    preserving the proportion of targeted customers in both datasets.

    Inputs:
    -------
        t: array-like, shape=(N, 1)
            Treatment array.
    Outputs:
    -------
        train_inds: array of bools
            Indices of the training set.
        valid_inds: array of bools
            Indices of the validation set.
    '''
    t_array = np.array(t)
    train_inds = np.zeros(len(t_array), dtype=bool)
    valid_inds = np.zeros(len(t_array), dtype=bool)
    values = np.unique(t_array)
    for value in values:
        value_inds = np.nonzero(t_array == value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion * len(value_inds))
        train_inds[value_inds[:n]] = True
        valid_inds[value_inds[n:]] = True
    return train_inds, valid_inds


def calculate_batch_size(batch_size, X_train):
    '''
    If batch_size is int than do nothing, else if batch_size is
    equal to None, set batch size to be of a size equal to the
    length of the training dataset.

    Inputs:
    -------
        batch_size: int or None
            Batch size.
        X_train: ndarray
            Array of consumer characteristics on which to
            perform training.
    Outputs:
    -------
        batch_size: int
            Batch size.
    '''
    if batch_size is None:
        batch_size = len(X_train)
    return batch_size


def plu_activation(input_value, alpha_=0.1, c=1):
    '''
    Apply PLU activation function on the input value.

    Inputs:
    -------
        input_value: Tensor
            An input tensor on which to apply PLU activation function.
        alpha: float
            First parameter of PLU function.
        c: float
            Second parameter of PLU function.

    Outputs:
    -------
            Transformed input values after applying PLU activation
            function.
    '''

    return tf.maximum(alpha_*(input_value+c) - c,
                      tf.minimum(alpha_*(input_value-c) + c,
                                 input_value))


def srelu_activation(input_value, scope_name):
    '''
    Apply S-shaped Rectified Linear activation function on the
    input value.

    Inputs:
    -------
        input_value: Tensor
            An input tensor on which to apply SReLU activation.
        scope_name: string
            Scope name.

    Outputs:
    -------
            Transformed input values after applying SReLU activation
            function.
    '''
    with tf.variable_scope(scope_name):
        t_right = tf.get_variable('t_right', input_value.get_shape()[-1],
                                  initializer=tf.constant_initializer(0.),
                                  dtype=tf.float32)
        a_right = tf.get_variable(
            'a_right', input_value.get_shape()[-1],
            initializer=tf.initializers.random_uniform(minval=0, maxval=1),
            dtype=tf.float32)
        t_left = tf.get_variable(
            't_left', input_value.get_shape()[-1],
            initializer=tf.initializers.random_uniform(minval=0, maxval=5),
            dtype=tf.float32)
        a_left = tf.get_variable('a_left', input_value.get_shape()[-1],
                                 initializer=tf.constant_initializer(1.),
                                 dtype=tf.float32)

    t_right_actual = t_left + tf.abs(t_right)
    y_left_and_center = t_left + tf.keras.activations.relu(
        input_value - t_left, a_left, t_right_actual - t_left
    )
    y_right = tf.nn.relu(input_value-t_right_actual) * a_right
    return y_left_and_center + y_right


def plotting_loss_functions(loss1, loss2=[], add_title=''):
    '''
    Plot the loss functions.

    Inputs:
    -------
        loss1: list of floats
            First list of recorded losses through epochs.
        loss2: list of floats
            Second list of recorded losses through epochs.
        add_title: string
            Addition to the title of the graph.
    '''
    plt.figure(figsize=(12, 5))
    plt.clf()
    plt.plot(range(len(loss1)), loss1, 'r-', lw=3)
    if early_stopping:
        plt.plot(range(len(loss2)), loss2, 'b-', lw=3)
        plt.legend(['loss on training set',
                    'loss on validation set'])
        plt.title('Loss on training and validation set' + add_title,
                  fontsize=14)
    else:
        plt.legend(['loss on training set'])
        plt.title('Loss on training set' + add_title, fontsize=14)
    plt.xlabel('Epoch number', fontsize=14)
    plt.ylabel('Loss', fontsize=14)


class NeuralNetwork():
    '''
    Create a neural network with specified properties.

    Inputs:
    -------
        hidden_layer_sizes: list of ints
            Length of the list defines the number of hidden layers.
            Entries of the list define the number of hidden units in
            each hidden layer.
        activation_functions: list of {'relu', 'lrelu', 'prelu',
                                       'srelu', 'plu', 'elu', 'none'}
            Activation function for each layer.
            Has to be of length len(hidden_layer_sizes) + 1.
        dropout_rates_train:  list of floats
            Dropout rate to be used during training for each layer.
            Has to be of length len(hidden_layer_sizes) + 1.
        batch_size: int
            Batch size.
        size_of_the_output: int
            Number of units in the output layer.
        nconsumer_characteristics: int
            Number of consumer characteristics.
        alpha: float
            Regularization strength parameter.
        r_par: float
            Mixing ratio of Ridge and Lasso regression.
            Has to be between 0 and 1.
        max_epochs_without_change: int
            Number of epochs with no improvement on the validation loss
            to wait before stopping the training.
        max_nepochs: int
            Maximum number of epochs for which NNs will be trained.
        optimizer: string
            Optimizer
        learning_rate: scalar
            Learning rate.
    '''
    def __init__(self, hidden_layer_sizes, activation_functions,
                 dropout_rates_train, batch_size, size_of_the_output,
                 nconsumer_characteristics=FLAGS.nconsumer_characteristics,
                 alpha=alpha, r_par=r,
                 max_epochs_without_change=max_epochs_without_change,
                 max_nepochs=max_nepochs, optimizer=optimizer,
                 learning_rate=learning_rate):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_functions = activation_functions
        self.dropout_rates_train = dropout_rates_train
        self.dropout_rates_test = [0 for i in dropout_rates_train]
        self.batch_size = batch_size
        self.size_of_the_output = size_of_the_output
        self.nconsumer_characteristics = nconsumer_characteristics
        self.alpha = alpha
        self.r_par = r_par
        self.max_epochs_without_change = max_epochs_without_change
        self.max_nepochs = max_nepochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def _fully_connected_layer_builder(self, input_data, hidden_layer_size,
                                       total_num_features, scope_name,
                                       activation, dropout_rate):
        '''
        Build a fully connected layer within the NN.

        Inputs:
        -------
            input_data: Tensor
                Output from the previous layer.
            hidden_layer_size: int
                Size of the current layer.
            total_num_features: int
                Number of units from the previous layer.
            scope_name: string
                Scope name.
            activation: {'relu', 'lrelu', 'prelu', 'srelu',
                         'plu', 'elu', 'none'}
                Activation function.
            dropout_rate: scalar
                Dropout rate. Has to be between 0 and 1.

        Outputs:
        -------
            hid_layer_activation: Tensor
                The hidden layer output.
        '''
        # Dropout:
        input_data = tf.contrib.layers.dropout(
            inputs=input_data, keep_prob=1-dropout_rate)

        # Creating weights and bias terms for our fully connected layer
        with tf.variable_scope(scope_name):
            weights = np.sqrt(2) * tf.get_variable(
                "weights",
                shape=[total_num_features, hidden_layer_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([hidden_layer_size]), name='biases')

        # Defining the fully connected neural network layer
        hid_layer_activation = tf.matmul(input_data, weights) + b

        if activation == 'relu':
            hid_layer_activation = tf.nn.relu(hid_layer_activation)
        elif activation == 'lrelu':
            hid_layer_activation = tf.nn.leaky_relu(hid_layer_activation,
                                                    alpha=0.2,
                                                    name='lrelu')
        elif activation == 'prelu':
            prelu_act = tf.keras.layers.PReLU()
            hid_layer_activation = prelu_act(hid_layer_activation)
        elif activation == 'srelu':
            hid_layer_activation = srelu_activation(hid_layer_activation,
                                                    scope_name)
        elif activation == 'plu':
            hid_layer_activation = plu_activation(hid_layer_activation)
        elif activation == 'elu':
            hid_layer_activation = tf.nn.elu(hid_layer_activation)
        elif activation == 'none':
            pass
        else:
            raise ValueError('Activation function not recognized! ' +
                             'Check the spelling.')
        return hid_layer_activation

    def _building_the_network(self, layer_input, dropout_rates):
        '''
        Build the whole fully connected NN.

        Inputs:
        -------
            layer_input: Tensor
                Input layer.
            dropout_rates: list of floats
                Dropout rate for each layer. Each entry has to
                be between 0 and 1. Has to be of length
                len(hidden_layer_sizes) + 1.

        Outputs:
        -------
            output_fc_layer: Tensor
                Output layer.
        '''
        hidden_layer_sizes_expand = self.hidden_layer_sizes + [
            self.size_of_the_output, self.nconsumer_characteristics]

        for i in range(len(self.hidden_layer_sizes) + 1):
            output_fc_layer = self._fully_connected_layer_builder(
                input_data=layer_input,
                hidden_layer_size=hidden_layer_sizes_expand[i],
                total_num_features=hidden_layer_sizes_expand[i-1],
                scope_name='l' + str(i+1),
                activation=self.activation_functions[i],
                dropout_rate=dropout_rates[i]
            )
            layer_input = output_fc_layer
        return output_fc_layer

    def _building_the_network_estimates_TE(self, input_data, t_var,
                                           y_var, dropout_rates):
        '''
        Build the neural network that estimates treatment
        coefficients.

        Inputs:
        -------
            layer_input: Tensor
                Input layer.
            t_var: Tensor
                Treatment
            y_var: Tensor
                Target variable
            dropout_rates: list of floats
                Dropout rate for each layer. Each entry has to
                be between 0 and 1. Has to be of length
                len(hidden_layer_sizes) + 1.

        Outputs:
        -------
            output: Tensor
                Treatment coefficients.
            loss: scalar
                Loss without regularization.
        '''

        output = self._building_the_network(input_data, dropout_rates)
        tau = output[:, 0:1]
        mu0 = output[:, 1:2]
        Y_predicted = tf.multiply(t_var, tau) + mu0

        # Mean squared error loss:
        loss = tf.losses.mean_squared_error(
            labels=y_var, predictions=Y_predicted)
        return output, loss

    def _building_the_network_estimates_PS(self, input_data,
                                           t_var, dropout_rates):
        '''
        Build the neural network that estimates propensity
        scores.

        Inputs:
        -------
            layer_input: Tensor
                Input layer.
            t_var: Tensor
                Treatment
            dropout_rates: list of floats
                Dropout rate for each layer. Each entry has to
                be between 0 and 1. Has to be of length
                len(hidden_layer_sizes) + 1.

        Outputs:
        -------
            output: Tensor
                Output of the NN.
            loss: scalar
                Loss without regularization.
        '''
        output = self._building_the_network(input_data, dropout_rates)

        # Calculating cross entropy loss
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.reshape(t_var, [-1, ]),
                logits=tf.reshape(output, [-1, ])))
        return output, loss

    def _calc_the_loss_with_reg(self, loss_before_regularization):
        '''
        Calculate loss with regularization.

        Inputs:
        -------
            loss_before_regularization: scalar
                Loss without regularization.

        Outputs:
        -------
            total_loss: float
                Loss with regularization.
        '''
        l1_l2_regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=self.alpha*self.r_par, scale_l2=self.alpha*(1-self.r_par))
        regularization_term = tf.contrib.layers.apply_regularization(
            l1_l2_regularizer, tf.trainable_variables(scope=r'l\d+/weights*'))

        total_loss = loss_before_regularization + regularization_term
        return total_loss

    def _optimize_the_loss_function(self, loss_with_regularization):
        '''
        Update the weights after one training step.

        Inputs:
        -------
            loss_with_regularization: scalar
                Loss with regularization.

        Outputs:
        -------
            train_step: Operation that updates the weights
        '''
        if self.optimizer == 'RMSProp':
            train_step = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(loss_with_regularization)
        if self.optimizer == 'GradientDescent':
            train_step = tf.train.GradientDescentOptimizer(
                self.learning_rate).minimize(loss_with_regularization)
        if self.optimizer == 'Adam':
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
                loss_with_regularization)
        return train_step

    def _create_minibatches(self, X, T, Y, rest, shuffle=False):
        '''
        Create mini-batches generator. Yields a mini-batch of batch_size
        length of consumer characteristics, X, treatments, T, target
        values, Y, and the indices of the remaining dataset.

        Inputs:
        -------
            X: ndarray, shape=(len(X_train), nconsumer_characteristics)
                Array of consumer characteristics.
            T: ndarray, shape=(len(X_train), 1)
                Treatment array.
            Y: ndarray, shape=(len(X_train), 1)
                Target value array.
            rest: list of ints
                Indices of the remaining array from the previous run.
            shuffle: bool
                If True, shuffle the array.
        Outputs:
        -------
            X[excerpt]: ndarray, shape=(batch_size,
                                        nconsumer_characteristics)
                Mini batch of consumer characteristics.
            T[excerpt]: ndarray, shape=(batch_size, 1)
                Mini batch of treatment values.
            Y[excerpt]: ndarray, shape=(batch_size, 1)
                Mini batch of target values.
            rest: list of ints
                Indices of the remaining array after current run.
        '''
        if shuffle:
            indices1 = np.arange(X.shape[0])
            np.random.shuffle(indices1)
            indices = np.array(rest + list(indices1))

        for start_idx in range(0, len(indices) - self.batch_size + 1,
                               self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx+self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx+self.batch_size)

            rest = list(indices[start_idx+self.batch_size:])
            yield X[excerpt], T[excerpt], Y[excerpt], rest

    def _training_the_NN(self, estimating_TE):
        '''
        Train a NN for max_nepochs or until early stopping criterion
        is met.

        Inputs:
        -------
            estimating_TE: bool
                Is neural network used for estimating treatment
                coefficients.

        Outputs:
        -------
            best_loss: float
                Minimum value of loss achieved on the validation set if
                train_proportion less than 1. Otherwise, loss achieved
                on the whole dataset during the last epoch.
            epoch_best: int
                Epoch at which minimum loss on validation set was
                achieved if train_proportion less than 1. Otherwise,
                equal to max_nepochs for which the NN is trained.
            output_best: ndarray
                Output of the NN at the epoch_best.
             total_nparameters: int
                 Number of neural network parameters
        '''

        # Placeholders
        x = tf.placeholder(
            tf.float32, shape=[None, FLAGS.nconsumer_characteristics])
        t = tf.placeholder(tf.float32, shape=[None, 1])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        dropout_rates = tf.placeholder(tf.float32, shape=[None])

        if estimating_TE:
            output, loss = self._building_the_network_estimates_TE(
                x, t, y, dropout_rates)
        else:
            output, loss = self._building_the_network_estimates_PS(
                x, t, dropout_rates)

        total_loss = self._calc_the_loss_with_reg(loss)

        train_step = self._optimize_the_loss_function(total_loss)

        sess = tf.InteractiveSession()
        # Initializing all variables
        sess.run(tf.global_variables_initializer())
        epoch_without_change = 0
        break_cond = False

        loss_train_list = []
        rest = []
        if early_stopping:
            loss_validation_list = []
            validation_loss_min = 10e6
            feed_dict_valid = {
                x: X_valid,
                t: T_valid,
                y: Y_valid,
                dropout_rates: dropout_rates_test
            }
        else:
            loss_whole_list = []

        feed_dict_total = {
            x: X,
            t: T_real,
            y: Y,
            dropout_rates: dropout_rates_test
        }

        for i in range(self.max_nepochs):
            if early_stopping:
                loss_valid = total_loss.eval(feed_dict=feed_dict_valid)
                loss_validation_list.append(loss_valid)
            else:
                loss_whole = total_loss.eval(feed_dict=feed_dict_total)
                loss_whole_list.append(loss_whole)

            if early_stopping:
                if validation_loss_min > loss_valid:
                    validation_loss_min = loss_valid
                    output_best = output.eval(feed_dict=feed_dict_total)
                    epoch_best = i
                    epoch_without_change = 0
                else:
                    epoch_without_change += 1

            s = 0
            for mini_batch in self._create_minibatches(
                    X_train, T_train, Y_train, rest, shuffle=True):

                x_batch, t_batch, y_batch, rest = mini_batch
                feed_dict_train = {
                    x: x_batch,
                    t: t_batch,
                    y: y_batch,
                    dropout_rates: self.dropout_rates_train
                }
                loss_train = sess.run(total_loss, feed_dict=feed_dict_train)
                if s == 0:
                    loss_train_list.append(loss_train)

                if epoch_without_change > max_epochs_without_change:
                    break_cond = True
                    break
                sess.run(train_step, feed_dict=feed_dict_train)
                s += 1

            if FLAGS.verbose:
                if i % 25 == 0:
                    if early_stopping:
                        print('%d epoch:' % i, 'loss on validation set:',
                              loss_valid)
                    else:
                        print('%d epoch:' % i, 'loss on whole set:',
                              loss_whole)

            # Check the stopping condition
            if break_cond:
                if FLAGS.verbose:
                    print('Training is finished! ', end='')
                    print('Best validation loss achieved at %d epoch'
                          % epoch_best)
                break

        if not early_stopping:
            output_best = output.eval(feed_dict=feed_dict_total)
            epoch_best = i + 1
            best_loss = loss_whole
            loss_list = loss_whole_list
        else:
            best_loss = validation_loss_min
            loss_list = loss_validation_list

        # Num of N parameters
        total_nparameters = np.sum([
            np.product([xi.value for xi in x.get_shape()])
            for x in tf.trainable_variables()])

        # Plotting loss functions
        if FLAGS.plot_true:
            if estimating_TE:
                add_title = ' - first NN'
            else:
                add_title = ' - second NN'

            # If train_proportion less than 1, than loss_list represents
            # list of losses on validation set after each epoch. If
            # train_proportion = 1, then it is list of losses on whole
            # dataset after each epoch.
            plotting_loss_functions(
                loss_train_list, loss_list, add_title)

        # Close tf.InteractiveSession
        sess.close()

        return best_loss, epoch_best, output_best, total_nparameters

    def training_the_NN_estimates_TE(self):
        '''
        Train a NN that estimates treatment coefficients for
        max_nepochs or until early stopping criterion is met.

        Outputs are the same as in ._training_the_NN function when
        estimating_TE argument is set to True.
        '''
        return self._training_the_NN(estimating_TE=True)

    def training_the_NN_estimates_PS(self):
        '''
        Train a NN that estimates propensity socres for
        max_nepochs or until early stopping criterion is met.

        Outputs are the same as in _training_the_NN function when
        estimating_TE argument is set to False.
        '''
        return self._training_the_NN(estimating_TE=False)


def influence_functions(mu0_pred, tau_pred, Y, T, prob_t_pred):
    '''
    Calculate the target value for each individual when treatment is
    0 or 1.

    Inputs:
    -------
        mu0_pred: ndarray, shape=(N, 1)
        tau_pred: ndarray, shape=(N, 1)
            Estimated conditional average treatment effect.
        Y: ndarray, shape=(N,)
            Target value array.
        T: ndarray, shape=(N,)
            Treatment array.
        prob_t_pred: ndarray, shape=(N,)
            Estimated propensity scores.
    Outputs:
    -------
        psi_0: ndarray, shape=(N, 1)
            Influence function for given x in case of no treatment.
        psi_1: ndarray, shape=(N, 1)
            Influence function for given x in case of treatment.
    '''
    first_part = (1-T) * (Y-mu0_pred)
    second_part = T * (Y-mu0_pred-tau_pred)

    if FLAGS.treatment == 'not_random':
        prob_t_pred[prob_t_pred < 0.0001] = 0.0001
        prob_t_pred[prob_t_pred > 0.9999] = 0.9999
        psi_0 = (first_part/(1-prob_t_pred)) + mu0_pred
        psi_1 = (second_part/prob_t_pred) + mu0_pred + tau_pred
    else:
        psi_0 = (first_part/(1-np.mean(T))) + mu0_pred
        psi_1 = (second_part/np.mean(T)) + mu0_pred + tau_pred
    return psi_0, psi_1


def update_model_comparison_file(name, model_info, cols):
    '''
    Update .csv file with new model results.

    Inputs:
    -------
        name: string
            File name. If the file does not already exist creates a
            new file. Otherwise appends new model results to the
            existing file.
        model_info: list
            Results of the current run.
        cols: list
            Names of columns within the .csv file.
            Has to be of the same length as model_info.
    '''
    if not os.path.isfile(name):
        df = pd.DataFrame(columns=cols)
        df.to_csv(name, index=False)
        print('File does not exist. Creating new file!')
    else:
        print('File already exists. Appending model run!')

    Model_comparison_Catalog_dataset = pd.read_csv(name)
    ind = len(Model_comparison_Catalog_dataset['Model number'])
    model_info[0][0] = ind
    df = pd.DataFrame(model_info, columns=cols)
    Model_comparison_Catalog_dataset = \
        Model_comparison_Catalog_dataset.append(df, ignore_index=True)
    Model_comparison_Catalog_dataset.to_csv(name, index=False)


def main():
    print('-------------------------------------------------------')
    print('Running Monte Carlo simulations for the following case:')
    print('* %s treatment' % FLAGS.treatment)
    print('* %s model' % FLAGS.model)
    print('* %s consumer characteristics'
          % FLAGS.nconsumer_characteristics, '\n')
    print('Using the following NN architectures:')
    print('First NN hidden layer sizes: ', hidden_layer_sizes)
    print('First NN hidden activations: ', activation_functions)
    print('First NN dropout rates: ', dropout_rates_train, '\n')

    print('Second NN hidden layer sizes: ', hidden_layer_sizes_treatment)
    print('Second NN hidden activations: ', activation_functions_treatment)
    print('Second NN dropout rates: ', dropout_rates_train_treatment)
    print('-------------------------------------------------------\n')

    count_in_interval = 0
    for _ in range(FLAGS.nsimulations):
        global X_train, T_train, Y_train, X_valid, \
            T_valid, Y_valid, X, T_real, Y
        # ---------------- Creating the fake dataset -----------------

        Y, X, mu0_real, tau_real, T_real, seed, prob_of_T,\
            tau_true_mean = FakeData().create_fake_data()

        # Setting the seed to prevent randomness:
        tf.set_random_seed(77)
        np.random.seed(61)

        # Splitting the dataset into training and validation set
        if early_stopping:
            train_inds, valid_inds = get_train_test_inds(T_real)
            T_train = T_real[train_inds]
            Y_train = Y[train_inds]
            X_train = X[train_inds]
            T_valid = T_real[valid_inds]
            Y_valid = Y[valid_inds]
            X_valid = X[valid_inds]
        else:
            T_train = T_real
            Y_train = Y
            X_train = X

        # Determining batch size
        batch_size_ = calculate_batch_size(batch_size, X_train)
        batch_size_t_ = calculate_batch_size(batch_size_t, X_train)

        # ------------- Building and training the first NN -----------

        if FLAGS.verbose:
            print('\nTraining of treatment coefficients neural network:')
        first_NN = NeuralNetwork(
            hidden_layer_sizes, activation_functions, dropout_rates_train,
            batch_size_, 2)

        MSE_best, epoch_best, betas_pred_best, total_nparameters = \
            first_NN.training_the_NN_estimates_TE()

        # ------------- Building and training the second NN ----------

        if FLAGS.treatment == 'not_random':
            # Reseting the graph
            tf.reset_default_graph()

            # Setting the seed to prevent randomness in our model:
            tf.set_random_seed(77)
            np.random.seed(61)

            if FLAGS.verbose:
                print('\nTraining of propensity score neural network:')
            second_NN = NeuralNetwork(
                hidden_layer_sizes_treatment, activation_functions_treatment,
                dropout_rates_train_treatment, batch_size_t_, 1)

            CE_best, epoch_best_t, treat_best, total_nparameters_t = \
                second_NN.training_the_NN_estimates_PS()

        # -------------------- Looking at the results ----------------

        betas_pred = betas_pred_best
        tau_pred = betas_pred[:, 0:1]
        mu0_pred = betas_pred[:, 1:]

        if FLAGS.treatment == 'not_random':
            prob_of_t_pred = 1 / (1 + np.exp(-treat_best))

        # Coefficients statistic
        mu0_mean_pred = np.mean(mu0_pred)
        std_mu0_pred = np.std(mu0_pred)
        tau_mean_pred = np.mean(tau_pred)
        std_tau_pred = np.std(tau_pred)

        mu0_mean_real = np.mean(mu0_real)
        std_mu0_real = np.std(mu0_real)
        tau_mean_real = np.mean(tau_real)
        std_tau_real = np.std(tau_real)

        if FLAGS.verbose:
            print('\n------------------ mu0 results ------------------')
            print(['Mean mu0_pred = %0.3f' % mu0_mean_pred,
                   'Std mu0_pred = %0.3f' % std_mu0_pred])
            print(['Mean mu0_real = %0.3f' % mu0_mean_real,
                   'Std mu0_real = %0.3f' % std_mu0_real], '\n')

            print('------------------ tau results ------------------')
            print(['Mean tau_pred = %0.3f' % tau_mean_pred,
                   'Std tau_pred = %0.3f' % std_tau_pred])
            print(['Mean tau_real = %0.3f' % tau_mean_real,
                   'Std tau_real = %0.3f' % std_tau_real], '\n')

            if FLAGS.treatment == 'not_random':
                print('------------------ t results ------------------')
                print('Mean prob_of_t_pred = %0.3f' % np.mean(prob_of_t_pred),
                      '\nMean prob_of_t_real = %0.3f\n' % np.mean(prob_of_T))

        total_nparameters_t = np.sum([
            np.product([xi.value for xi in x.get_shape()])
            for x in tf.trainable_variables()
        ])

        if FLAGS.treatment == 'not_random':
            psi_0, psi_1 = influence_functions(mu0_pred, tau_pred, Y,
                                               T_real, prob_of_t_pred)
        else:
            psi_0, psi_1 = influence_functions(mu0_pred, tau_pred, Y,
                                               T_real, prob_t_pred=None)

        # Calculating confidence interval for average treatment effect
        mean_diff_psi1_psi0 = np.mean(psi_1 - psi_0)
        std_diff_psi1_psi0 = np.std(psi_1 - psi_0)
        CI_upper_bound = (
            mean_diff_psi1_psi0 + 1.96*std_diff_psi1_psi0/np.sqrt(nconsumers)
        )
        CI_lower_bound = (
            mean_diff_psi1_psi0 - 1.96*std_diff_psi1_psi0/np.sqrt(nconsumers)
        )

        in_95_conf_int = CI_lower_bound < tau_true_mean < CI_upper_bound

        print('is tau_true_mean in interval:', in_95_conf_int)
        print('CI lower and upper bound are: (%0.3f, %0.3f)'
              % (CI_lower_bound, CI_upper_bound))
        if in_95_conf_int:
            count_in_interval += 1

        Y_pred = mu0_pred + tau_pred*T_real

        # ----------------- Saving the results! ----------------------
        name = 'Results_{}_model_{}{}_{}_{}.csv'.format(
                FLAGS.model, FLAGS.architecture, FLAGS.treatment,
                nconsumers, FLAGS.nconsumer_characteristics)

        if FLAGS.treatment == 'random':
            parameters_dict = {
                'nconsumer_characteristics': FLAGS.nconsumer_characteristics,
                'treatment': FLAGS.treatment, 'model': FLAGS.model,
                'architecture': FLAGS.architecture,
                'hidden_layer_sizes': hidden_layer_sizes,
                'dropout_rates_train': dropout_rates_train,
                'activation_functions': activation_functions,
                'nconsumers': nconsumers,
                'train_proportion': train_proportion,
                'max_nepochs': max_nepochs,
                'max_epochs_without_change': max_epochs_without_change,
                'early_stopping': early_stopping, 'optimizer': optimizer,
                'learning_rate': learning_rate, 'batch_size': batch_size,
                'alpha': alpha, 'r': r}

            cols = [
                'Model number', 'seed', 'best_epoch', 'total_nparameters',
                'Loss best', 'Mean mu0_pred', 'Std mu0_pred', 'Mean mu0_real',
                'Std mu0_real', 'Mean tau_pred', 'Std tau_pred',
                'Mean tau_real', 'Std tau_real', 'tau_true_mean',
                'Y_real_mean', 'Y_pred_mean', 'CI_lower_bound',
                'CI_upper_bound', 'psi_0_mean', 'psi_1_mean',
                'mean_diff_psi_1_psi_0', 'std_diff_psi_1_psi_0', 'in_interval',
                'model_parameters_dict']

            model_info = [[
                0, seed, epoch_best, total_nparameters,
                MSE_best, mu0_mean_pred, std_mu0_pred, np.mean(mu0_real),
                std_mu0_real, tau_mean_pred, std_tau_pred, tau_mean_real,
                std_tau_real, tau_true_mean, np.mean(Y), np.mean(Y_pred),
                CI_lower_bound, CI_upper_bound, np.mean(psi_0), np.mean(psi_1),
                mean_diff_psi1_psi0, std_diff_psi1_psi0, in_95_conf_int,
                parameters_dict]]

        else:
            parameters_dict = {
                'nconsumer_characteristics': FLAGS.nconsumer_characteristics,
                'treatment': FLAGS.treatment, 'model': FLAGS.model,
                'architecture': FLAGS.architecture,
                'hidden_layer_sizes': hidden_layer_sizes,
                'dropout_rates_train': dropout_rates_train,
                'activation_functions': activation_functions,
                'hidden_layer_sizes_treatment': hidden_layer_sizes_treatment,
                'activation_functions_treatment': activation_functions_treatment,
                'dropout_rates_train_treatment': dropout_rates_train_treatment,
                'nconsumers': nconsumers, 'train_proportion': train_proportion,
                'max_nepochs': max_nepochs,
                'max_epochs_without_change': max_epochs_without_change,
                'early_stopping': early_stopping, 'optimizer': optimizer,
                'learning_rate': learning_rate, 'batch_size': batch_size,
                'batch_size_t': batch_size_t, 'alpha': alpha, 'r': r}

            cols = [
                'Model number', 'seed', 'best_epoch', 'best_epoch_t',
                'total_nparameters', 'total_nparameters_t', 'Loss best',
                'Loss best_treatment', 'Mean mu0_pred', 'Std mu0_pred',
                'Mean mu0_real', 'Std mu0_real', 'Mean tau_pred',
                'Std tau_pred', 'Mean tau_real', 'Std tau_real',
                'tau_true_mean', 'Mean_prob_t_pred', 'Mean prob_of_t_real',
                'mean_T_real', 'Y_real_mean', 'Y_pred_mean', 'CI_lower_bound',
                'CI_upper_bound', 'psi_0_mean', 'psi_1_mean',
                'mean_diff_psi_1_psi_0', 'std_diff_psi_1_psi_0', 'in_interval',
                'model_parameters_dict']

            model_info = [[
                0, seed, epoch_best, epoch_best_t, total_nparameters,
                total_nparameters_t, MSE_best, CE_best,
                mu0_mean_pred, std_mu0_pred, mu0_mean_real, std_mu0_real,
                tau_mean_pred, std_tau_pred, tau_mean_real, std_tau_real,
                tau_true_mean, np.mean(prob_of_t_pred), np.mean(prob_of_T),
                np.mean(T_real), np.mean(Y), np.mean(Y_pred), CI_lower_bound,
                CI_upper_bound, np.mean(psi_0), np.mean(psi_1),
                mean_diff_psi1_psi0, std_diff_psi1_psi0, in_95_conf_int,
                parameters_dict]]

        if FLAGS.update:
            update_model_comparison_file(name, model_info, cols)
        tf.reset_default_graph()  # restarting a NN graph
        print('\n')

    print('%d out of %d simulations contain tau_true_mean in the CI'
          % (count_in_interval, FLAGS.nsimulations))

    # Print running time
    running_time()

    if os.path.exists(name):
        model_data = pd.read_csv(name)
        print(model_data)
        print('%d out of %d simulations contain tau_true_mean in the CI'
              % (np.sum(model_data['in_interval']), len(model_data)))
        print('Name of the file:', name)
    else:
        print('File `' + name + '` is not yet created.')

    # Plot all the graphs
    plt.show()


if __name__ == "__main__":
    main()
