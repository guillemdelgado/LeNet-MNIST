import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


N = 64
D = 1000
H = 100
experiment_name = 'simple_Example'

# Simple example with 3 layers
x = tf.placeholder(tf.float32, [N, D])
y = tf.placeholder(tf.float32, [N, D])

init = tf.contrib.layers.xavier_initializer()
h_1 = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init)
h_2 = tf.layers.dense(inputs=h_1, units=H, activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h_2, units=D, kernel_initializer=init)

loss = tf.losses.mean_squared_error(y_pred, y)
acc, acc_op = tf.contrib.metrics.streaming_accuracy(y_pred, y)

learning_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #Set to 1 in order to calculate the accuracy
    values = {x: np.random.randint(1, size=(N, D)),
              y: np.random.randint(1, size=(N, D))}
    for t in range(50):
        loss_val = sess.run([loss, updates], feed_dict=values)
        #Accuracy only works for booleans.
        accuracy_val = sess.run([acc, acc_op], feed_dict=values)

    print('Loss value: {}'.format(loss_val))
    print('Accuracy value: {}'.format(accuracy_val))





