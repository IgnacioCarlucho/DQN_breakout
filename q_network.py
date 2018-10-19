
import tensorflow as tf
import numpy as np

EPSILON = 0.1

class Network(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, SIZE_FRAME, action_dim, learning_rate, device):
        self.sess = sess
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.currentState = -1.
        self.device = device
        self.SIZE_FRAME = SIZE_FRAME
        
        # Q network
        self.inputs, self.out, self.saver, self.X = self.create_q_network('q')
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_saver, self.target_X = self.create_q_network('q_target')

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        #self.update_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]
        self.update_target_network_params = [oldp.assign(p) for p, oldp in zip(self.network_params, self.target_network_params)] # vf_params, vf_old_params


        #self.global_step = tf.train.get_or_create_global_step()
        #epsilon_decay = tf.train.polynomial_decay(EPSILON, self.global_step, 1e5, 0.01, power=0.0)


        with tf.device(self.device):

            self.target_q_t = tf.placeholder(tf.float32, [None, 1], name='target_q')
            self.action = tf.placeholder(tf.int32, [None, 1], name='action')
            # obtain the q scores of the selected action
            self.action_one_hot = tf.one_hot(self.action, self.a_dim, name='action_one_hot')
            self.q_acted_0 = self.out * tf.squeeze(self.action_one_hot)
            self.q_acted = tf.reduce_sum(self.q_acted_0, reduction_indices=1, name='q_acted')

            #self.action_one_hot = tf.one_hot(self.action, self.a_dim, 1.0, 0.0, name='action_one_hot')
            #self.q_acted = tf.reduce_sum(self.out * self.action_one_hot, reduction_indices=1, name='q_acted')
            #self.q_reduced = tf.reduce_mean(self.q_acted, axis=1)

            #self.delta = tf.subtract(tf.stop_gradient(self.target_q_t), q_acted)
            #self.loss = self.clipped_error(self.delta)
            #self.loss = tf.reduce_mean(self.clipped_error(self.delta), name='loss')
            self.target_final = tf.squeeze(self.target_q_t)
            self.delta = tf.subtract(tf.stop_gradient(self.target_final), self.q_acted)
            self.loss = tf.reduce_mean(self.clipped_error(self.delta), name='loss')

            #self.loss = tf.losses.huber_loss(self.target_final, self.q_acted, reduction=tf.losses.Reduction.MEAN)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = self.optimizer.compute_gradients(self.loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5.), var)
            self.optimize = self.optimizer.apply_gradients(gradients)
            


        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_q_network(self, scope):

        with tf.device(self.device):
            with tf.variable_scope(scope): 
                '''
                W_conv1 = tf.Variable(tf.random_normal([8,8,4,32]))
                W_conv2 = tf.Variable(tf.random_normal([4,4,32,64]))
                W_conv3 = tf.Variable(tf.random_normal([3,3,64,64]))
                
                #W_fc = tf.Variable(tf.random_normal([7*7*64,512])*np.sqrt(2./(7*7*64)))
                #W_out = tf.Variable( tf.random_normal([512, self.a_dim])*np.sqrt(1./(512)) )

                W_fc = tf.Variable(np.random.uniform(size=(7*7*64,512),low= -0.0003, high=0.0003 ).astype(np.float32))
                W_out = tf.Variable(np.random.uniform(size=(512,self.a_dim),low= -0.0003, high=0.0003 ).astype(np.float32))

                b_conv1 = tf.Variable(tf.random_normal([32]))
                b_conv2 = tf.Variable(tf.random_normal([64]))
                b_conv3 = tf.Variable(tf.random_normal([64]))
                b_fc = tf.Variable(np.zeros([512]).astype(np.float32))
                b_out = tf.Variable(np.zeros([self.a_dim]).astype(np.float32))
                # input
                '''
                #b_out = tf.Variable(np.zeros([self.a_dim]).astype(np.float32))
                #W_out = tf.Variable(np.random.uniform(size=(512,self.a_dim),low= -0.0003, high=0.0003 ).astype(np.float32))
                stateInput = tf.placeholder(tf.uint8, shape=[None,self.SIZE_FRAME,self.SIZE_FRAME,4], name='stateInput') # 84,84,4
                
                '''
                # first cnn lay
                conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1,4) + b_conv1)
                #conv1 = maxpool2d(conv1)
                # second cnn layer
                conv2 = tf.nn.relu(self.conv2d(conv1, W_conv2,2) + b_conv2)
                #conv2 = maxpool2d(conv2)
                # third cnn layer
                conv3 = tf.nn.relu(self.conv2d(conv2, W_conv3,1) + b_conv3)
                #conv2 = maxpool2d(conv2)
                '''
                X = tf.to_float(stateInput) / 255.0
                #X = tf.reshape(stateInput,shape=[-1, self.SIZE_FRAME, self.SIZE_FRAME, 4])
                conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.leaky_relu)
                conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.leaky_relu)
                conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.leaky_relu)

                # fully connected layer

                flattened = tf.contrib.layers.flatten(conv3)
                fc = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.leaky_relu)
                out = tf.contrib.layers.fully_connected(fc, self.a_dim, activation_fn=None)
                #out = tf.contrib.layers.fully_connected(fc, self.a_dim, activation_fn=None)
                # out = tf.matmul(fc, W_out)+ b_out
                #fc = tf.nn.relu(tf.matmul(flattened,W_fc)+ b_fc)
                # ouput layer
                #out = tf.matmul(fc, W_out)+ b_out
                #output = tf.nn.softmax(tf.matmul(fc, W_out)+ b_out)
                '''
                # Xavier initialization: 
                regularizer = tf.contrib.layers.l2_regularizer(0.1)
                W_conv1 = tf.get_variable("W_conv1_a", shape=[4,4,4,32],regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b_conv1 = tf.Variable(tf.zeros([32]))
                W_conv2 = tf.get_variable("W_conv2_a", shape=[4,4,32,32],regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b_conv2 = tf.Variable(tf.zeros([32]))
                W_conv3 = tf.get_variable("W_conv3_a", shape=[3,3,32,32],regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b_conv3 = tf.Variable(tf.zeros([32]))
                W_fc1 = tf.get_variable("W_fc1_a", shape=[1568,200],regularizer = regularizer, initializer=tf.contrib.layers.xavier_initializer())
                b_fc1 = tf.Variable(tf.zeros([200]))
                W_fc2 = tf.Variable(np.random.uniform(size=(200,self.a_dim),low= -0.0003, high=0.0003 ).astype(np.float32))
                b_fc2 = tf.Variable(tf.zeros([self.a_dim])) 
                
                # input layer

                stateInput = tf.placeholder(tf.float32, shape=[None,self.SIZE_FRAME,self.SIZE_FRAME,4]) # 84,84,4
                # COnv layers
                h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
                h_conv3_flat = tf.reshape(h_conv3,[-1,1568])
                h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
                out = tf.matmul(h_fc1,W_fc2) + b_fc2
                '''
        saver = tf.train.Saver()
        return stateInput, out, saver, conv3

        
    def train(self, actions, target_q_t, inputs):
        with tf.device(self.device):
            return self.sess.run([self.loss, self.optimize], feed_dict={
                self.action: actions,
                self.target_q_t: target_q_t,
                self.inputs: inputs
            })

    def train_v2(self, actions, target_q_t, inputs):
        with tf.device(self.device):
            return self.sess.run([self.target_final, self.q_acted, self.delta, self.loss, self.optimize], feed_dict={
                self.action: actions,
                self.target_q_t: target_q_t,
                self.inputs: inputs
            })

    def predict(self, inputs):
        with tf.device(self.device):
            return self.sess.run([self.out, self.X], feed_dict={
                self.inputs: inputs
            })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    
    def update_target_network(self):
        with tf.device(self.device):
            self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def save(self):
        self.saver.save(self.sess,'./model.ckpt')
        self.target_saver.save(self.sess,'./model_target.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file: model")

    
    def recover(self):
        self.saver.restore(self.sess,'./model.ckpt')
        self.target_saver.restore(self.sess,'./model_target.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    
    def conv2d(self,x, W, stride):
        with tf.device(self.device):
            return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def clipped_error(self,x):
      # Huber loss
      try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
      except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)