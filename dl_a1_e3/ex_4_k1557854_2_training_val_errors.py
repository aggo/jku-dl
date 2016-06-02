import numpy as np
from matplotlib.pyplot import savefig


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


def backward_pass(activations, w0, w1, w2, y):
    '''
    Calculate the derivatives of the weights of the neural network.
    The hidden layers have tanh activations, the output layer a softmax activation.

    activations: list of the activations of each layer during the forward pass
                 in the following order: [x, h0, h1, o]
    w0: input->hidden0 weights   [n_hidden0 * n_inputs]
    w1: input->hidden1 weights   [n_hidden1 * n_hidden0]
    w2: input->output weights    [n_outputs * n_hidden1]
    y:  labels as 1hot-encoding  [n_samples * n_outputs]

    Returns:
    A list that contains the derivatives of the biases and
    the weights of each layer in the following order:
    [dw0, db0, dw1, db1, dw2, db2]
    '''

    inputs, activations_hidden_layer1, activations_hidden_layer2, output = activations

    error_output_layer = output - y  # difference between the output of the network and the true value (aka delta3)
    derivatives_of_cost_fct_wrt_hidden_layer2_weights = np.dot(error_output_layer.T, activations_hidden_layer2)
    derivatives_of_cost_fct_wrt_bias_on_hidden_layer2 = output.sum(0)

    error_hidden_layer_2 = np.dot(error_output_layer, w2) * (
    1 - activations_hidden_layer2 * activations_hidden_layer2)  # tanh' = 1 - tanh^2
    derivatives_of_cost_fct_wrt_hidden_layer1_weights = np.dot(error_hidden_layer_2.T, activations_hidden_layer1)
    derivatives_of_cost_fct_wrt_bias_on_hidden_layer1 = error_hidden_layer_2.sum(0)

    error_hidden_layer_1 = np.dot(error_hidden_layer_2, w1) * (
    1 - activations_hidden_layer1 * activations_hidden_layer1)
    derivatives_of_cost_fct_wrt_input_weights = np.dot(error_hidden_layer_1.T, inputs)
    derivatives_of_cost_fct_wrt_bias_on_input_layer = error_hidden_layer_1.sum(0)

    dw0 = derivatives_of_cost_fct_wrt_input_weights
    db0 = derivatives_of_cost_fct_wrt_bias_on_input_layer
    dw1 = derivatives_of_cost_fct_wrt_hidden_layer1_weights
    db1 = derivatives_of_cost_fct_wrt_bias_on_hidden_layer1
    dw2 = derivatives_of_cost_fct_wrt_hidden_layer2_weights
    db2 = derivatives_of_cost_fct_wrt_bias_on_hidden_layer2

    return [dw0, db0, dw1, db1, dw2, db2]


def softmax_loss(w0, b0, w1, b1, w2, b2, x, y):
    _, _, _, out = forward_pass(w0, b0, w1, b1, w2, b2, x)
    err = -np.sum(y * np.log(out+1e-9))/x.shape[0]  # / x.shape[0]
    return err

def read_data(filename):
    from bokeh.models import pd
    df = pd.read_csv(
        filepath_or_buffer=filename,
        header=None,
        sep=' ')
    data = np.array(df.ix[:, :].values)
    return data

def read_labels(filename):
    from bokeh.models import pd
    df = pd.read_csv(
        filepath_or_buffer=filename,
        header=None,
        sep=' ')
    data = np.array(df.ix[:, :].values)
    return data

def split_data(inputs, true_labels):
    size = len(inputs)
    train_inputs = inputs[:1/2*size, :]
    val_inputs = inputs[1/2*size:3/4*size,:]
    test_inputs = inputs[3/4*size:,:]
    train_labels = true_labels[:1/2*size,:]
    val_labels =true_labels[1/2*size:3/4*size,:]
    test_labels = true_labels[3/4*size:,:]
    return train_inputs,  test_inputs, train_labels,  test_labels, val_inputs, val_labels


def plot_train_val_error_vs_epoch(errors_on_val, errors_on_train, title):
    import matplotlib.pyplot as plt

    epochs_val = len(errors_on_val)
    epochs_train = len(errors_on_train)

    epochs = np.linspace(0, epochs_val, epochs_val)
    epochs2 = np.linspace(0, epochs_train, epochs_train)

    fig = plt.figure()
    plt.interactive(False)
    plt.xlabel("Epochs")
    plt.ylabel("Error")

    print(epochs_train)
    print(errors_on_train)
    plt.plot(epochs2, errors_on_train, 'r',epochs, errors_on_val,'b')
    plt.legend(["Training error", "Validation error"])
    plt.plot()

    fig.savefig(title + ".png", bbox_inches='tight')


def gradient_descent(learning_rate, hidden_u_l1, hidden_u_l2,
                     max_epochs, convergence_condition, min_error):
    epoch = 0

    n_hidden0 = hidden_u_l1
    n_hidden1 = hidden_u_l2

    w0 = np.random.uniform(-0.1, +0.1, size=(n_hidden0, train_inputs.shape[1]))
    b0 = np.zeros((1, n_hidden0))

    w1 = np.random.uniform(-0.1, +0.1, size=(n_hidden1, n_hidden0))
    b1 = np.zeros((1, n_hidden1))

    w2 = np.random.uniform(-0.1, +0.1, size=(train_labels.shape[1], n_hidden1))
    b2 = np.zeros((1, train_labels.shape[1]))
    error_on_val = 1000000;  # some very large value
    errors_on_train = []
    errors_on_val = []

    while True:
        # forward propagation
        activations = forward_pass(w0, b0, w1, b1, w2, b2, train_inputs)

        # back propagation
        dw0, db0, dw1, db1, dw2, db2 = backward_pass(activations, w0, w1, w2, train_labels)

        # weights and biases update
        w0 -= learning_rate * dw0
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b0 -= learning_rate * b0
        b1 -= learning_rate * b1
        b2 -= learning_rate * b2

        # compute and save loss on val and train
        prev_error = error_on_val
        error_on_train = softmax_loss(w0,  b0, w1,b1, w2,  b2, train_inputs, train_labels)
        errors_on_train.append(error_on_train)
        error_on_val = softmax_loss(w0,  b0, w1,b1, w2,  b2, val_inputs, val_labels)
        errors_on_val.append(error_on_val)
        title = "LR:_{0}_H1:_{1}_H2_{2}_EP_{3}:_Error_on_train:_{4}_on val:_{5}".format(learning_rate, hidden_u_l1, hidden_u_l2, epoch, error_on_train, error_on_val)
        print(title)

        # increase epoch and break if too many epocs
        epoch += 1
        if epoch > max_epochs:
            break
        if (np.abs(prev_error-error_on_val))<convergence_condition:
            break
        if (error_on_val<min_error):
            break

    return [w0, b0, w1, b1, w2, b2, epoch, errors_on_train, errors_on_val]

def grid_search():
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    hidden_units_first_layer = [8, 10, 12, 15, 20]
    hidden_units_second_layer = [8, 10, 12, 15, 20]

    # learning_rates = [ 0.001]
    # hidden_units_first_layer = [10]
    # hidden_units_second_layer = [10]
    error_min = 100000

    for learning_rate in learning_rates:
        for hidden_unit_first_layer in hidden_units_first_layer:
            for hidden_unit_second_layer in hidden_units_second_layer:
                [w0, b0, w1, b1, w2, b2, epochs, _, _] = gradient_descent(learning_rate, hidden_unit_first_layer, hidden_unit_second_layer,
                                 max_epochs, convergence_condition, min_error_conv)
                error_on_val = softmax_loss(w0, b0, w1, b1, w2, b2, val_inputs, val_labels)
                if error_on_val<error_min:
                    error_min = error_on_val
                    [min_lr, min_h1, min_h2, min_epochs] = [learning_rate, hidden_unit_first_layer, hidden_unit_second_layer, epochs]

    return [min_lr, min_h1, min_h2, min_epochs, error_on_val]

inputs = read_data("mnist.csv")
true_labels = read_labels("mnist-labels.csv")
train_inputs,  test_inputs, train_labels,  test_labels, val_inputs, val_labels = split_data(inputs, true_labels)
max_epochs = 200
convergence_condition = 0.001
min_error_conv = 0.01
# find the values of the params which minimize the error
[min_lr, min_h1, min_h2, min_epochs, error_on_val] = grid_search()

print("Optimal LR:")
print(min_lr)
print("Optimal nr. of neurons on hidden layer 1:")
print(min_h1)
print("Optimal nr. of neurons on hidden layer 2:")
print(min_h2)
print("Min epochos")
print(min_epochs)
print("Error on validation set:")
print(error_on_val)


# then run gradient descent again on train and on val to plot the error vs epoch
[w0, b0, w1, b1, w2, b2, epoch, errors_on_train, errors_on_val] = gradient_descent(min_lr, min_h1, min_h2, min_epochs, convergence_condition, min_error_conv)

# plot error vs epoch
plot_train_val_error_vs_epoch(errors_on_val, errors_on_train, "Error_rate_by_the_number_of_epochs")
