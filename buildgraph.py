import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name = 'W')
b = tf.Variable(tf.zeros([1]), name = 'b')
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name = "train")

# Before starting, initialize the variables
init = tf.initialize_all_variables()

# Save the graph
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '.', 'graph.pb', as_text=False)
tf.train.write_graph(sess.graph_def, '.', 'text.pb')
