from mlp import MLP
from prepro import prepro


X_, y_ = prepro('/home/frl/PycharmProjects/test/data/')
layer_1 = MLP(X=X_, y=y_)
layer_1.train()

