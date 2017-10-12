import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime as dt


class toy_data(object):
    def __init__(self):
        self.data = np.matmul(np.linspace(
            1, 10, 10).reshape(10, 1), np.ones((1, 100)))

    def NA_example(self):
        rand_example = self.data[:, np.random.randint(
            0, 99)].reshape(self.data.shape[0], 1)
        mask = np.random.rand(rand_example.shape[0], 1) < 0.6
        return rand_example * mask

    def sample(self):
        return self.data[:, 0:10]


# class toy_data(object):
#     def __init__(self):
#         self.mu = 4
#         self.sigma = 0.5
#         # self.range = 8
#         self.x_n = 10

#     def sample(self, x_m=10):
#         samples = np.ndarray((self.x_n, x_m))
#         for i in range(x_m):
#             samples[:,i] = np.random.normal(self.mu, self.sigma, self.x_n)
#         samples.sort(0)
#         return samples

#     def NA_example(self):
#         sample = np.random.normal(self.mu, self.sigma, self.x_n).reshape((self.x_n, 1))
#         mask = np.random.rand(self.x_n, 1) < 0.8
#         return sample * mask

# np.histogram
# plt.plot(toy_data().NA_example())
# toy_data().sample()


def optimizer(loss, var_list):
    learning_rate = 0.009
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


tf.reset_default_graph()

def linear_layer(input, output_dim, scope=None, batch_norm=True, activation=tf.nn.relu):
    # input dims should be (x_n, x_m)
    # output dims are (hidden_units, x_m)
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable("W", [output_dim, input.get_shape()[0]], initializer=tf.contrib.layers.xavier_initializer())
        # tf.summary.histogram('W', W)
        if batch_norm:
            Z_ = tf.matmul(W, input)
            # tf.summary.histogram('Z_', Z_)
            Z = tf.layers.batch_normalization(Z_, axis=0)
        else:
            b = tf.get_variable("b", [output_dim, 1], initializer=tf.zeros_initializer())
            Z = tf.add(tf.matmul(W, input), b)
        # tf.summary.histogram('Z', Z)
        if activation:
            return activation(Z)
        else:
            return Z

def generator_network(input):
    with tf.variable_scope("Generator"):        
        A1 = linear_layer(input, 10, scope="L1", batch_norm=True, activation=tf.nn.relu)
        # tf.summary.histogram('activation_1', A1)
        A2 = linear_layer(A1, 10, scope="L2", batch_norm=True, activation=tf.nn.relu)
        # tf.summary.histogram('activation_2', A2)
        Output = linear_layer(A2, 10, scope="Out", batch_norm=False, activation=False)
        # tf.summary.tensor_summary('Output', Output)
        # tf.summary.histogram('Output', Output)
        return Output

def discriminator_network(input, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        A1 = linear_layer(input, 20, scope="L1", batch_norm=True, activation=tf.nn.relu)
        A2 = linear_layer(A1, 20, scope="L2", batch_norm=True, activation=tf.nn.relu)
        # A3 = linear_layer(A2, 20, scope="L3", batch_norm=True, activation=tf.nn.relu)
        Output = linear_layer(A2, 10, scope="Out", batch_norm=True, activation=tf.nn.sigmoid)
        return Output

# GENERATOR
with tf.variable_scope("Generator_input"):
    G_In = tf.placeholder(dtype="float", shape=(10, 1))
G_Ou = generator_network(G_In)

# DISCRIMINATOR
with tf.variable_scope("Discriminator_input"):
    D_In = tf.placeholder(dtype="float", shape=(10, 10))

D1_Tr = discriminator_network(D_In)
D2_Fa = discriminator_network(G_Ou, reuse=True)

with tf.variable_scope("Descriminator_loss"):
    loss_D = tf.reduce_mean(-tf.log(D1_Tr) - tf.log(1 - D2_Fa))

with tf.variable_scope("Generator_loss"):
    loss_G = tf.reduce_mean(-tf.log(D2_Fa)) + tf.reduce_mean(tf.square((G_In - G_Ou) * tf.cast(tf.cast(G_In, tf.bool), tf.float32)))

with tf.variable_scope("Generator_perfomance"):
    difference = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.constant([1,2,3,4,5,6,7,8,9,10], dtype=tf.float32, shape=[10,1]), G_Ou)))

tf.summary.scalar('Generator_difference', difference)
tf.summary.scalar('Discriminator_loss', loss_D)
tf.summary.scalar('Generator_loss', loss_G)

vars = tf.trainable_variables()
D_params = [v for v in vars if v.name.startswith('Discriminator/')]
G_params = [v for v in vars if v.name.startswith('Generator/')]

opt_D = optimizer(loss_D, D_params)
opt_G = optimizer(loss_G, G_params)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("/tmp/tensorflow/logs/GAN/"+dt.now().strftime('%Y-%m-%d-%H-%M'), sess.graph)
    costsG = []
    costsD = []
    for step in range(3000 + 1):
        # update discriminator
        X = toy_data().sample()
        F = toy_data().NA_example()
        summary, D_step_loss, _ = sess.run([merged, loss_D, opt_D], {D_In: X, G_In: F})
        writer.add_summary(summary, step)
        costsD.append(D_step_loss)

        # update generator
        #F = toy_data().NA_example()
        G_step_loss, _ = sess.run([loss_G, opt_G], {G_In: F})
        costsG.append(G_step_loss)
        if step % 100 == 0:
            print('{}: {:.4f}\t{:.4f}'.format(step, D_step_loss, G_step_loss))

    writer.close()
    T = toy_data().NA_example()
    print(T)
    print(sess.run(G_Ou, {G_In: T}))

    # plot the costD
    plt.plot(np.squeeze(costsD))
    plt.ylabel('Discriminator cost')
    plt.xlabel('iterations (per tens)')
    #plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # plot the costG
    plt.plot(np.squeeze(costsG))
    plt.ylabel('Generator cost')
    plt.xlabel('iterations (per tens)')
    #plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    save_path = saver.save(sess, "/tmp/tensorflow/models/GAN/model_"+dt.now().strftime('%Y-%m-%d-%H-%M')+".ckpt")
    print("Model saved in file: %s" % save_path)


with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/tensorflow/models/GAN/model_"+".ckpt")
    print("Model restored.")
    # Check the values of the variables
    T = np.array([1, 2, 0, 0, 5, 0, 0, 0, 0, 10]).reshape(10, 1)
    print(T)
    print(sess.run(G_Ou, {G_In: T}))


# c = tf.constant([[1.,2.,0.,4.]])
# b = tf.cast(tf.cast(c, tf.bool), tf.int32)
# r = tf.multiply(tf.constant([[1,2,10,4]]), b)

# with tf.Session() as sess:
#     print(sess.run(r))

# x = tf.constant
