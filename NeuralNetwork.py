import numpy as np

class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        # self.weights_input_to_hidden.shape = (2,56)
        # inputs.shape = (56,1)
        # hidden_inputs.shape = (2,1)
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer
        # self.weights_hidden_to_output.shape = (1,2)
        # hidden_outputs.shape = (2,1)
        # final_inputs.shape = (1,1)
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###
        error = targets - final_outputs
        # error gradient of output_error = (y-y_hat) f'(h_k)
        output_error = error # since this is a scalar

        # error at output layer
        # del_o = (y_k - yhat_k) * f'(h_k) # since the activation is just the identity
        # del_o = error * 1
        del_err_output = error

        # error at hidden layer j
        # TODO had to take the transpose of the calculated gradient
        del_err_hidden = np.dot(self.weights_hidden_to_output.T, output_error) # * (hidden_outputs * (1-hidden_outputs))

        # TODO this was weird dimensionaly as well
        delta_w_h_o = self.lr * del_err_output * hidden_outputs.T
        delta_w_i_o = self.lr * del_err_hidden * inputs.T

        self.weights_hidden_to_output += delta_w_h_o
        self.weights_input_to_hidden += delta_w_i_o


    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs

