# robust-neural-network
A reusable neural network library with configurable hidden layer structure written in tensorflow.

The robby.py file is a library whose usage has been demonstrated in Main.py file. It creates a tensorflow based neural network
for solving classification problems with structure as specified by the user.

For instance if user passes an array [5,2,8,3], the generated neural network will have these many nodes in each layer.
It adds an additional output layer with a single node on its own for generating softmax probabilities.
The neurons use relu function and have shown excellent results with scikit-learn datasets.

