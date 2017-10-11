import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class toy_data(object):
    def __init__(self):
        self.data = np.matmul(np.linspace(
            1, 10, 10).reshape(10, 1), np.ones((1, 100)))

    def NA_example(self):
        rand_example = self.data[:, np.random.randint(
            0, 99)].reshape(self.data.shape[0], 1)
        mask = np.random.rand(rand_example.shape[0], 1) < 0.8
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
    learning_rate = 0.0001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


tf.reset_default_graph()

# GENERATOR
G_In = tf.placeholder(dtype="float", shape=(10, 1))
G_W1 = tf.get_variable(
    "G_W1", [10, 10], initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.get_variable("G_b1", [10, 1], initializer=tf.zeros_initializer())
G_W2 = tf.get_variable(
    "G_W2", [10, 10], initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.get_variable("G_b2", [10, 1], initializer=tf.zeros_initializer())
G_W3 = tf.get_variable(
    "G_W3", [10, 10], initializer=tf.contrib.layers.xavier_initializer())
G_b3 = tf.get_variable("G_b3", [10, 1], initializer=tf.zeros_initializer())

G_Z1 = tf.add(tf.matmul(G_W1, G_In), G_b1)
G_Z1 = tf.layers.batch_normalization(G_Z1, axis=0)
G_A1 = tf.nn.relu(G_Z1)
G_Z2 = tf.add(tf.matmul(G_W2, G_A1), G_b2)
G_Z2 = tf.layers.batch_normalization(G_Z2, axis=0)
G_A2 = tf.nn.relu(G_Z2)
G_Ou = tf.add(tf.matmul(G_W3, G_A2), G_b3)

# DISCRIMINATOR
D_In = tf.placeholder(dtype="float", shape=(10, 10))
D_W1 = tf.get_variable(
    "D_W1", [10, 10], initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.get_variable("D_b1", [10, 1], initializer=tf.zeros_initializer())
D_W2 = tf.get_variable(
    "D_W2", [10, 10], initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.get_variable("D_b2", [10, 1], initializer=tf.zeros_initializer())
D_W3 = tf.get_variable(
    "D_W3", [1, 10], initializer=tf.contrib.layers.xavier_initializer())
D_b3 = tf.get_variable("D_b3", [1, 1], initializer=tf.zeros_initializer())

D1_Z1 = tf.add(tf.matmul(D_W1, D_In), D_b1)
D1_Z1 = tf.layers.batch_normalization(D1_Z1, axis=0)
D1_A1 = tf.nn.relu(D1_Z1)
D1_Z2 = tf.add(tf.matmul(D_W2, D1_A1), D_b2)
D1_Z2 = tf.layers.batch_normalization(D1_Z2, axis=0)
D1_A2 = tf.nn.relu(D1_Z2)
D1_Z3 = tf.add(tf.matmul(D_W3, D1_A2), D_b3)
D1_Z3 = tf.layers.batch_normalization(D1_Z3, axis=0)
D1_Tr = tf.nn.sigmoid(D1_Z3)

D2_Z1 = tf.add(tf.matmul(D_W1, G_Ou), D_b1)
D2_Z1 = tf.layers.batch_normalization(D2_Z1, axis=0)
D2_A1 = tf.nn.relu(D2_Z1)
D2_Z2 = tf.add(tf.matmul(D_W2, D2_A1), D_b2)
D2_Z2 = tf.layers.batch_normalization(D2_Z2, axis=0)
D2_A2 = tf.nn.relu(D2_Z2)
D2_Z3 = tf.add(tf.matmul(D_W3, D2_A2), D_b3)
D2_Z3 = tf.layers.batch_normalization(D2_Z3, axis=0)
D2_Fa = tf.nn.sigmoid(D2_Z3)

loss_D = tf.reduce_mean(-tf.log(D1_Tr) - tf.log(1 - D2_Fa))
loss_G = tf.reduce_mean(-tf.log(D2_Fa)) + tf.reduce_mean(
    tf.square((G_In - G_Ou) * tf.cast(tf.cast(G_In, tf.bool), tf.float32)))

loss_D_summary = tf.summary.scalar('Discriminator loss', loss_D)
loss_G_summary = tf.summary.scalar('Generator loss', loss_G)

vars = tf.trainable_variables()
D_params = [v for v in vars if v.name.startswith('D_')]
G_params = [v for v in vars if v.name.startswith('G_')]

opt_D = optimizer(loss_D, D_params)
opt_G = optimizer(loss_G, G_params)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("/tmp/log4/", sess.graph)
    costsG = []
    costsD = []
    for step in range(20000 + 1):
        # update discriminator
        X = toy_data().sample()
        F = toy_data().NA_example()
        loss_D_sum, D_step_loss, _ = sess.run([loss_D_summary, loss_D, opt_D], {D_In: X, G_In: F})
        writer.add_summary(loss_D_sum, step)
        costsD.append(D_step_loss)

        # update generator
        #F = toy_data().NA_example()
        loss_G_sum, G_step_loss, _ = sess.run([loss_G_summary, loss_G, opt_G], {G_In: F})
        writer.add_summary(loss_G_sum, step)

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

    save_path = saver.save(sess, "/tmp/model4.ckpt")
    print("Model saved in file: %s" % save_path)


with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model4.ckpt")
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
