# Author: Satwik Bhattamishra

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

def denoising_autoencoder(hidden_units= 1000, prob=0.4):

	learning_rate = 0.005
	batch_size = 50
	n_epochs = 300

	x = tf.placeholder(tf.float32, [None, 784], name='x')

	x_ = tf.add(x ,np.random.normal(loc= 0.0, scale=prob , size= (batch_size, 784) ))

	n_inp = 784
	n_out = hidden_units

	W = tf.Variable(tf.random_uniform([n_inp, n_out], -1.0 / np.sqrt(n_inp), 1.0 / np.sqrt(n_inp)) ,dtype=tf.float32 )
	b = tf.Variable(tf.truncated_normal([n_out], dtype=tf.float32))
	W_ = tf.Variable(tf.random_uniform([n_out, n_inp], -1.0 / np.sqrt(n_inp), 1.0 / np.sqrt(n_inp)) ,dtype=tf.float32 )
	b_ = tf.Variable(tf.truncated_normal([n_inp], dtype=tf.float32))

	z = tf.nn.sigmoid(tf.matmul(x_ , W) + b)

	y = tf.nn.sigmoid(tf.matmul(z , W_) + b_)


	cost = tf.reduce_mean(-tf.reduce_sum(x * tf.log(tf.clip_by_value(y ,1e-10,1.0))) )

	print("Fetching Data...")
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	mean_img = np.mean(mnist.train.images, axis=0)

	print("Starting Session...")

	saver = tf.train.Saver()

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# saver.restore(sess, "./weights/da.cpkt")


	for i in range(n_epochs):
		avg_cost = 0
		batches= mnist.train.num_examples // batch_size

		for batch_i in range(batches):
			batch_xs, _ = mnist.train.next_batch(batch_size)
			# train = np.array([img - mean_img for img in batch_xs])
			_, ce = sess.run([optimizer, cost], feed_dict={x : batch_xs})

			avg_cost += ce / batches

		print(i, avg_cost)




if __name__ == '__main__':
	denoising_autoencoder()
