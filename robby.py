import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

W = []
B = []
Z = []
A = []
curr_class = 0

nodes = []
n_classes = 0
n_samples = 0
n_layers = 0
n_feat = 0
graphs = {}

X = tf.placeholder(tf.float64)  # n_feat * n_samples
Y = tf.placeholder(tf.float64)  # 1 * n_samples


def log(tag, msg):
    print('LOG ::: ' + str(tag) + '  :::  ' + str(msg))


def initialize():
    global W, B, Z, A, nodes, X, Y

    l = len(nodes)

    W = [None for i in range(0, l)]
    B = [None for i in range(0, l)]
    Z = [None for i in range(0, l)]
    A = [None for i in range(0, l)]


def onevall(x):
    global curr_class
    return 1.0 if int(x) == int(curr_class) else 0.0


def scale_util(x):
    scaler = StandardScaler()
    scaler.fit(np.transpose(x))
    return np.transpose(scaler.transform(np.transpose(x)))


def train(x=[], y=[], n_iter=0, descent=0.0, hidden=[]):
    global n_classes, graphs, curr_class, nodes, X, Y, n_samples

    x = np.double(x)
    y = np.double(y)

    x = scale_util(x)

    n_classes = len(np.unique(y))
    graphs = {}

    f = np.vectorize(onevall)

    hidden.append(1)
    nodes = hidden
    n_layers = len(nodes)
    log('n_layers', n_layers)

    probs = None
    labels = None

    q_dict = {}

    for i in range(0, n_classes):

        curr_class = i

        ytemp = f(np.array(y))
        log('ytemp', ytemp)

        res = train_one_net(x, ytemp, n_iter=n_iter, descent=descent, hidden=hidden)

        temp = res['A'][n_layers - 1]

        probs = temp if i == 0 else tf.concat([probs, temp], axis=0)
        labels = tf.Variable(ytemp) if i == 0 else tf.concat([labels, tf.Variable(ytemp)],
                                                             axis=0)  # doesn't go n_classes * n_samples

        for j in range(0, n_layers):
            q_dict[str(i) + 'W' + str(j)] = res['W'][j]
            q_dict[str(i) + 'A' + str(j)] = res['A'][j]
            q_dict[str(i) + 'Z' + str(j)] = res['Z'][j]
            q_dict[str(i) + 'B' + str(j)] = res['B'][j]

    labels = tf.transpose(tf.reshape(labels, shape=(n_classes, n_samples)))  # n_samples * n_classes
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(probs), labels=labels))

    train_step = tf.train.AdamOptimizer(descent).minimize(loss)

    q_dict['loss'] = loss
    q_dict['probs'] = probs
    q_dict['train_step'] = train_step
    q_dict['labels'] = labels

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    results = None
    for i in range(0, n_iter):
        results = sess.run(q_dict, feed_dict={X: x, Y: y})
        log('result loss', results['loss'])

    log('labels', results['labels'])
    graphs = results


def train_one_net(x=[], y=[], n_iter=0, descent=0.0, hidden=[]):
    global nodes, W, B, Z, A, graphs, n_layers, X, Y, n_samples, n_feat

    initialize()

    n_feat = x.shape[0]
    n_samples = x.shape[1]

    n_prev = n_feat

    for i in range(0, len(nodes)):

        n_curr = nodes[i]

        W[i] = tf.Variable(np.random.rand(n_prev, n_curr), dtype=tf.float64)  # n_prev * n_curr
        B[i] = tf.Variable(np.random.rand(n_curr, 1), dtype=tf.float64)  # n_curr * 1

        if i == 0:
            Z[i] = tf.matmul(tf.transpose(W[i]), X) + B[i]  # n_curr * n_samples
        else:
            Z[i] = tf.matmul(tf.transpose(W[i]), A[i - 1]) + B[i]  # n_curr * n_samples

        A[i] = tf.nn.relu(Z[i])  # n_curr * n_samples

        n_prev = n_curr

    ret = dict()

    ret['W'] = W
    ret['Z'] = Z
    ret['A'] = A
    ret['B'] = B

    return ret


def predict(x=[]):
    global graphs, n_layers, n_feat, nodes, n_classes

    x = np.double(x)
    x = scale_util(x)

    ans = []
    n_layers = len(nodes)

    pred = []
    for i in range(0, n_classes):

        log('A' + str(i), graphs[str(i) + 'A' + str(n_layers - 1)])

        n_prev = n_feat

        a = tf.Variable(np.array([]), dtype=tf.float64)  # n_curr * n_samples   # 1 * n_samples for last layer
        z = tf.Variable(np.array([]), dtype=tf.float64)  # n_curr * n_samples
        log('n_layers', n_layers)

        for j in range(0, n_layers):
            w = tf.Variable(graphs[str(i) + 'W' + str(j)], dtype=tf.float64)  # n_prev * n_curr
            b = tf.Variable(graphs[str(i) + 'B' + str(j)], dtype=tf.float64)  # n_curr * 1

            n_curr = nodes[j]

            a = tf.Variable(x, dtype=tf.float64) if j == 0 else a
            z = tf.matmul(tf.transpose(w), a) + b  # n_curr * n_samples

            a = tf.nn.relu(z)  # n_curr * n_samples

            n_prev = n_curr

        ans.append(a)  # (n_classes * n_samples)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    ans = sess.run([ans])
    log('sess ans', ans)

    ans = np.transpose(np.reshape(ans, (n_classes, x.shape[1])))

    log('pred ans', ans)

    ret = np.array([0 for i in range(0, ans.shape[0])])
    for i in range(0, ans.shape[0]):

        mi = 0
        mv = ans[i][0]
        for j in range(1, n_classes):
            mi = j if mv < ans[i][j] else mi
            mv = max(mv, ans[i][j])
        ret[i] = mi

    log('ret', ret)
    return ret


def run_test():
    corpus = load_iris()

    x = np.transpose(corpus.data)  # Input dims should be n_features * n_samples
    y = corpus.target  # Input dims should be 1 * n_samples

    log('trainx.shape', x.shape)
    log('trainy.shape', y.shape)

    train(x, y, n_iter=600, descent=0.0012, hidden=[5, 2])

    pred = predict(x)
    log('pred', pred)
    log('y', y)

    check = (pred == y)

    hits = float(np.sum(check))
    acc = hits / len(check) * 100

    log('acc%', acc)


run_test()