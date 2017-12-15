from sklearn.datasets import load_digits
import numpy as np
import robby as nn

corpus = load_digits()

x = np.transpose( corpus.images.reshape( corpus.images.shape[0], corpus.images.shape[1] * corpus.images.shape[2] ) )  # Input dims of features should be n_features * n_samples
y = corpus.target  # Input dims of classes should be 1 * n_samples

nn.log('trainx.shape', x.shape)
nn.log('trainy.shape', y.shape)

nn.train(x, y, n_iter=600, descent=0.0012, hidden=[5, 2])  # Neural network automatically adds an additional output layer of 1 node

pred = nn.predict(x)
nn.log('pred', pred)
nn.log('y', y)

check = (pred == y)

hits = float(np.sum(check))
acc = hits / len(check) * 100

nn.log('acc%', acc)