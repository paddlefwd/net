import numpy as np
import six
import net
import unittest
import matplotlib.pyplot as plt

class TestNet(unittest.TestCase):
    def test_linear_regression(self):
        nvis = 2
        nout = 1

        # Simple linear regression.
        model = net.Layer(activation=net.identity,
                nvis=nvis, nhid=nout, learning_rate=0.1)

        np.random.seed(17)
        n_examples = 1000
        x = np.random.normal(size=(n_examples,nvis))
        y = (x[:, 0] - 2*x[:, 1]).reshape((n_examples,nout))

        for epoch in six.moves.range(1000):
            y_hat = model.forward(x)

            assert y.shape == y_hat.shape

            # Hard-coded cost function.
            mse = np.sum((y - y_hat)**2)/len(x)

            # Hard-coded gradient computation.
            grad = np.dot(x.T, y_hat - y)

            # Updating parameters outside of the layer.
            model.W = model.W - model.lr * grad/float(len(x))

        self.assertTrue(np.allclose(y, y_hat))

    def test_logistic_regression(self):
        nvis = 2
        nclasses = 1

        # Simple logistic regression.
        model = net.Layer(activation=net.sigmoid,
                nvis=nvis, nhid=nclasses, learning_rate=0.01)

        np.random.seed(17)
        n_examples = 1000

        def make_data(n_examples, nvis):
            x = np.zeros((n_examples, nvis))
            x[0:n_examples/2] = np.random.normal(-3, 1, size=(n_examples/2,nvis))
            x[n_examples/2:] = np.random.normal(3, 1, size=(n_examples/2,nvis))
            x += np.random.uniform(-1, 1, size=x.shape)

            y = np.zeros((n_examples, 1))
            y[0:n_examples/2] = 0
            y[n_examples/2:] = 1

            idx = np.arange(len(x))
            perm = np.random.permutation(idx)

            x = x[perm]
            y = y[perm]

            return x,y

        def cost(y, y_hat, n):
            return np.sum(-y*np.log(y_hat - (1-y)*np.log(1-y_hat)))/float(n)

        x_train, y_train = make_data(n_examples, nvis)
        x_valid, y_valid= make_data(n_examples, nvis)

        """
        colors = ['red' if y == 0 else 'green' for y in y_train]
        plt.scatter(x_train[:, 0], x_train[:, 1], c=colors)
        plt.show(block=False)
        """

        for epoch in six.moves.range(50):
            # Training set.
            y_hat_train = model.forward(x_train)
            assert y_train.shape == y_hat_train.shape
            train_cost = cost(y_train, y_hat_train, len(x_train))

            # Validation set.
            y_hat_valid = model.forward(x_valid)
            assert y_valid.shape == y_hat_valid.shape
            valid_cost = cost(y_valid, y_hat_valid, len(x_valid))

            # Hard-coded gradient computation.
            grad = np.dot(x_train.T, y_hat_train - y_train)

            # Updating parameters outside of the layer.
            model.W = model.W - model.lr * grad/float(len(x_train))

        self.assertTrue(np.all(np.round(y_hat_train) == y_train))

if __name__ == '__main__':
    unittest.main()
