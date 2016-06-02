import numpy as np


def softmax(z):
    m = z.max(1)[:, None]  # improves numerical stability of softmax
    e = np.exp(z - m)
    return e / e.sum(1)[:, None]


def forward_pass(w0, b0, w1, b1, w2, b2, x):
    '''
    Calculate the output of a network with 2 hidden layers
    with tanh hidden and softmax output activation.

    w0: input->hidden0 weights
    b0: bias of hidden0 layer
    w1: hidden0->hidden1 weights
    b1: bias of hidden1 layer
    w2: hidden1->output weights
    b2: bias of output layer
    x: input matrix [n_samples * n_features]

    Returns a list that contains the activations of each layer
    '''

    # inputs to the first layer
    inputs_hidden_layer1 = np.dot(x, w0.T) + b0
    activations_hidden_layer1 = np.tanh(inputs_hidden_layer1)

    inputs_hidden_layer2 = np.dot(activations_hidden_layer1, w1.T) + b1
    activations_hidden_layer2 = np.tanh(inputs_hidden_layer2)

    inputs_outputLayer = np.dot(activations_hidden_layer2, w2.T) + b2
    output = softmax(inputs_outputLayer)

    h0 = activations_hidden_layer1
    h1 = activations_hidden_layer2
    o = output

    return [x, h0, h1, o]


def backward_pass(activations, w0, w1, w2, y, llambda):
    '''
    Calculate the derivatives of the weights of the neural network.
    The hidden layers have tanh activations, the output layer a softmax activation.

    activations: list of the activations of each layer during the forward pass
                 in the following order: [x, h0, h1, o]
    w0: input->hidden0 weights   [n_hidden0 * n_inputs]
    w1: input->hidden1 weights   [n_hidden1 * n_hidden0]
    w2: input->output weights    [n_outputs * n_hidden1]
    y:  labels as 1hot-encoding  [n_samples * n_outputs]
    llambda =

    Returns:
    A list that contains the derivatives of the biases and
    the weights of each layer in the following order:
    [dw0, db0, dw1, db1, dw2, db2]
    '''

    inputs, activations_hidden_layer1, activations_hidden_layer2, output = activations

    error_output_layer = output - y  # difference between the output of the network and the true value (aka delta3)
    derivatives_of_cost_fct_wrt_hidden_layer2_weights = np.dot(error_output_layer.T, activations_hidden_layer2)+\
        +llambda * np.sum(activations_hidden_layer2)
        # the regularization (l2 weight decay) term = lambda * sum(over all weights on this layer) weight^2

    derivatives_of_cost_fct_wrt_bias_on_hidden_layer2 = output.sum(0)

    error_hidden_layer_2 = np.dot(error_output_layer, w2) * (
    1 - activations_hidden_layer2 * activations_hidden_layer2)  # tanh' = 1 - tanh^2
    derivatives_of_cost_fct_wrt_hidden_layer1_weights = np.dot(error_hidden_layer_2.T, activations_hidden_layer1)+\
                                                        llambda * np.sum(activations_hidden_layer1)
    derivatives_of_cost_fct_wrt_bias_on_hidden_layer1 = error_hidden_layer_2.sum(0)

    error_hidden_layer_1 = np.dot(error_hidden_layer_2, w1) * (
    1 - activations_hidden_layer1 * activations_hidden_layer1)
    derivatives_of_cost_fct_wrt_input_weights = np.dot(error_hidden_layer_1.T, inputs)+ \
                                                llambda * np.sum(w0)
    derivatives_of_cost_fct_wrt_bias_on_input_layer = error_hidden_layer_1.sum(0)

    dw0 = derivatives_of_cost_fct_wrt_input_weights
    db0 = derivatives_of_cost_fct_wrt_bias_on_input_layer
    dw1 = derivatives_of_cost_fct_wrt_hidden_layer1_weights
    db1 = derivatives_of_cost_fct_wrt_bias_on_hidden_layer1
    dw2 = derivatives_of_cost_fct_wrt_hidden_layer2_weights
    db2 = derivatives_of_cost_fct_wrt_bias_on_hidden_layer2

    return [dw0, db0, dw1, db1, dw2, db2]


def softmax_loss(w0, b0, w1, b1, w2, b2, x, y, llambda):
    _, _, _, out = forward_pass(w0, b0, w1, b1, w2, b2, x)
    err = -np.sum(y * np.log(out))+llambda*np.sum(w2**2)+llambda*np.sum(w1**2)+llambda*np.sum(w0**2) # weight decay: sum of weights
    # as seen here: https://msdn.microsoft.com/en-us/magazine/dn904675.aspx
    return err


## Generate some random data to test the network with
n_samples = 30
n_features = 20
n_outputs = 5
n_hidden0 = 20
n_hidden1 = 10

llambdas = [0.01, 0.1, 1, 10, 100]


x = np.random.normal(size=(n_samples, n_features))
y = np.random.multinomial(1, [1.0 / n_outputs] * n_outputs, size=n_samples).astype(np.float64)

w0 = np.random.uniform(-0.1, +0.1, size=(n_hidden0, x.shape[1]))
b0 = np.zeros((1, n_hidden0))

w1 = np.random.uniform(-0.1, +0.1, size=(n_hidden1, n_hidden0))
b1 = np.zeros((1, n_hidden1))

w2 = np.random.uniform(-0.1, +0.1, size=(y.shape[1], n_hidden1))
b2 = np.zeros((1, y.shape[1]))


from scipy.optimize import approx_fprime

for llambda in llambdas:
    print("Lambda = {0}".format(llambda))
    eta = 1e-4

    nw0 = approx_fprime(w0.ravel(), lambda w: softmax_loss(w.reshape(w0.shape), b0, w1, b1, w2, b2, x, y, llambda), eta)
    nb0 = approx_fprime(b0.ravel(), lambda b: softmax_loss(w0, b.reshape(b0.shape), w1, b1, w2, b2, x, y, llambda), eta)

    nw1 = approx_fprime(w1.ravel(), lambda w: softmax_loss(w0, b0, w.reshape(w1.shape), b1, w2, b2, x, y, llambda), eta)
    nb1 = approx_fprime(b1.ravel(), lambda b: softmax_loss(w0, b0, w1, b.reshape(b1.shape), w2, b2, x, y, llambda), eta)

    nw2 = approx_fprime(w2.ravel(), lambda w: softmax_loss(w0, b0, w1, b1, w.reshape(w2.shape), b2, x, y, llambda), eta)
    nb2 = approx_fprime(b2.ravel(), lambda b: softmax_loss(w0, b0, w1, b1, w2, b.reshape(b2.shape), x, y, llambda), eta)

    nw0 = nw0.reshape(w0.shape)
    nb0 = nb0.reshape(b0.shape)
    nw1 = nw1.reshape(w1.shape)
    nb1 = nb1.reshape(b1.shape)
    nw2 = nw2.reshape(w2.shape)
    nb2 = nb2.reshape(b2.shape)

    activations = forward_pass(w0, b0, w1, b1, w2, b2, x)
    dw0, db0, dw1, db1, dw2, db2 = backward_pass(activations, w0, w1, w2, y, llambda)

    # All of these numbers should be smaller than 1e-4
    # Otherwise something might have gone wrong
    print("Max Deviation layer 0: ", (dw0 - nw0).max(), '\t/\t', (db0 - nb0).max())
    print("Max Deviation layer 1: ", (dw1 - nw1).max(), '\t/\t', (db1 - nb1).max())
    print("Max Deviation layer 2: ", (dw2 - nw2).max(), '\t/\t', (db2 - nb2).max())
