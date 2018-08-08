import numpy as np


class MLP(object):
    def __init__(self, X, y, eta=0.01, n_iter=100, batch_size=60000):
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.w_ = np.zeros((10, X.shape[1]))
        self.b_ = np.zeros(10)
        self.errors_ = []
        self.data = X.reshape((len(y)//self.batch_size, self.batch_size, 784))
        self.label = y.reshape((len(y)//self.batch_size, self.batch_size, 10))
        self.batch_index = 0
        self.batch_data = self.data[self.batch_index]
        self.batch_label = self.label[self.batch_index]
        self.clock = 0

    def get_next_batch(self):
        if self.batch_index < 60000 / self.batch_size - 1:
            self.batch_index += 1
            self.batch_data = self.data[self.batch_index]
            self.batch_label = self.label[self.batch_index]

    def restart(self):
        self.batch_index = 0

    def j_linear(self):
        #[l_k,i,j], [l_k,i]
        J_w = np.zeros((self.batch_size, 10, 7840))
        J_b = np.zeros((self.batch_size, 10, 10))
        for m in range(self.batch_size):
            for k in range(10):
                J_w[m][k][k * 784: (k+1) * 784] = self.batch_data[m]
                J_b[m][k][k] += 1
        self.linear_J_w = J_w
        self.linear_J_b = J_b

    def v_linear(self):
        V = np.zeros((self.batch_size, 10))
        for m in range(self.batch_size):
            V[m] = np.dot(self.w_, self.batch_data[m].reshape((784, 1))).reshape(10) + self.b_
        self.linear_V = V

    def j_softmax(self):
        #[s_k,i,j],[s_k,i]
        J_w = np.zeros((self.batch_size, 10, 7840))
        J_b = np.zeros((self.batch_size, 10, 10))
        for m in range(self.batch_size):
            v = np.exp(self.linear_V[m])
            j = np.diag(v) / np.sum(v) - np.dot(v.reshape((10, 1)), v.reshape((1, 10))) / (np.sum(v) * np.sum(v))
            J_w[m] = np.dot(j, self.linear_J_w[m])
            J_b[m] = np.dot(j, self.linear_J_b[m])
        self.softmax_J_w = J_w
        self.softmax_J_b = J_b

    def v_softmax(self):
        V = np.zeros((self.batch_size, 10))
        for m in range(self.batch_size):
            V[m] = np.exp(self.linear_V[m]) / np.sum(np.exp(self.linear_V[m]))
        self.softmax_V = V

    def j_loss(self):
        J_w = np.zeros((self.batch_size, 7840))
        J_b = np.zeros((self.batch_size, 10))
        for m in range(self.batch_size):
            j = np.zeros((1, 10))
            j[0][np.argmax(self.batch_label[m])] = -1.0 / self.softmax_V[m][np.argmax(self.batch_label[m])]
            J_w[m] = np.dot(j, self.softmax_J_w[m]).reshape(7840)
            J_b[m] = np.dot(j, self.softmax_J_b[m]).reshape(10)
        self.loss_J_w = J_w
        self.loss_J_b = J_b

    def v_loss(self):
        V = 0
        for m in range(self.batch_size):
            V += -np.log(self.softmax_V[m][np.argmax(self.batch_label[m])])
        self.loss_V = V

    def calculus(self):
        self.j_linear()
        self.v_linear()
        self.j_softmax()
        self.v_softmax()
        self.j_loss()
        self.v_loss()
        G_w = np.zeros(7840)
        G_b = np.zeros(10)

        for m in range(self.batch_size):
            G_w += self.loss_J_w[m]
            G_b += self.loss_J_b[m]
        self.G_w = G_w.reshape((10, 784))
        self.G_b = G_b

    def apply_grad(self):
        self.w_ += -self.eta * self.G_w
        self.b_ += -self.eta * self.G_b

    def predict(self):
        return np.argmax(self.softmax_V, 1)

    def output(self):
        count = 0.0
        self.clock += 1
        p = self.predict()
        for m in range(self.batch_size):
            if p[m] == np.argmax(self.batch_label[m]):
                count += 1.0
        print("step: ", self.clock, "precise: ", count / self.batch_size, "loss: ", self.loss_V)

    def train(self):
        for i in range(self.n_iter):
            for j in range(60000 // self.batch_size):
            #for j in range(2):
                self.calculus()
                self.apply_grad()
                self.get_next_batch()
                self.output()
                #print(self.linear_V[0], self.softmax_V[0])
            self.restart()












