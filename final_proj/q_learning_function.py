import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import math
import pickle
import timeit
import matplotlib.pyplot as plt
import itertools
from cs231n.data_utils import load_CIFAR10

MAX_LAYER = 10
NUM_ACTION = 16
NUM_MODEL = 500

# Action space 
def get_action_space():
    action_space = [] 

    # Softmax
    action_0 = {'type': 'SM',
                'num_output': 10}
    action_space.append(action_0) 

    # Convolution
    for filter_size in [1, 3, 5]:
        for num_filter in [64, 128, 256, 512]:
            action = {'type': 'C', 
                      'filter_size': filter_size, 
                      'stride': 1, 
                      'num_filter': num_filter}
            action_space.append(action)

    # Pooling
    action_13 = {'type': 'P', 
                'filter_size': 5, 
                'stride': 3}
    action_14 = {'type': 'P', 
                'filter_size': 3, 
                'stride': 2}
    action_15 = {'type': 'P', 
                'filter_size': 2, 
                'stride': 2}
    action_space.append(action_13)
    action_space.append(action_14)
    action_space.append(action_15)
    return action_space

action_space = get_action_space()

def get_CIFAR10_data(num_training=20000, num_validation=1000, num_test=5000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=128, print_every=100,
              training=None, plot_losses=False, verbose=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0

    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0 and verbose:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        
        if verbose:
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct

def action_to_hex(action):
    return hex(action)[2:]
        
def hex_to_action(hex_str):
    if hex_str is '$':
        return 16
    return int('0x' + hex_str, 16)

def is_pool(i):
    if 13 <= i <= 15:
        return True
    return False

def sample_network(epsilon=1.0, Q=None):
    state = '$'
    action_sequence = []
    prev_action = 16
    
    for layer in range(MAX_LAYER):  
        
        # Last layer softmax
        if layer == MAX_LAYER - 1:
            i = 0
        
        # e-greedy 
        elif np.random.rand() < epsilon:
            if is_pool(prev_action):
                i = np.random.randint(0, 13)
            else:
                i = np.random.randint(0, NUM_ACTION)
        else:
            if is_pool(prev_action):
                i = np.argmax(Q[prev_action,:12])
            else:
                i = np.argmax(Q[prev_action, :]) 
                
        state += action_to_hex(i)
        action_sequence.append(action_space[i])
        prev_action = i
        
        if i == 0:
            break
            
    return state, action_sequence

def build_model(action_sequence, X, y):
    inputs = X
    for i in range(len(action_sequence)):
        layer = action_sequence[i]
        if layer['type'] is 'SM':  
            break
        
        elif layer['type'] is 'C':
            outputs = layers.conv2d(inputs=inputs,
                                    num_outputs=layer['num_filter'],
                                    kernel_size=layer['filter_size'],
                                    stride=layer['stride'],
                                    padding='same',
                                    activation_fn=tf.nn.relu)
        
        elif layer['type'] is 'P':
            outputs = layers.max_pool2d(inputs=inputs, 
                                        kernel_size=layer['filter_size'], 
                                        stride=layer['stride'],
                                        padding='same')
        
        inputs = outputs 
    
    flat = layers.flatten(inputs)
    outputs = layers.fully_connected(inputs=flat,
                                     num_outputs=10,
                                     activation_fn=None)
    return outputs

def update_Q(Q, state, reward, learning_rate, discount):
    S = state[:-1]
    A = state[1:]
    s = hex_to_action(S[-1])
    a = hex_to_action(A[-1])
    Q[s, a] = (1 - learning_rate)*Q[s, a] + learning_rate * reward
    for i in range(len(state) - 3, -1, -1):
        s = hex_to_action(S[i])
        a = hex_to_action(A[i])
        next_s = a
        if is_pool(next_s):
            Q[s, a] = (1 - learning_rate)*Q[s, a] + learning_rate * discount * np.max(Q[next_s, :12]) 
        else:
            Q[s, a] = (1 - learning_rate)*Q[s, a] + learning_rate * discount * np.max(Q[next_s, :])
    return Q

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

replay = {}
epsilon = 1.0
Q = 0.5*np.ones([NUM_ACTION + 1, NUM_ACTION])

for i_episode in range(NUM_MODEL):
    print('==========Episode %d==========' % (i_episode))
    state, action_sequence = sample_network(epsilon, Q)
    
    # Train the model
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    y_out = build_model(action_sequence, X, y)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out)
    mean_loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_step = optimizer.minimize(mean_loss)

    with get_session() as sess:
        with tf.device("/gpu:0") as dev:
            sess.run(tf.global_variables_initializer())
            print('Training ', state)
            loss, accuracy = run_model(sess,y_out,mean_loss,X_train,y_train,10,64,100,train_step, verbose=True)
            print('Testing')
            _, reward = run_model(sess,y_out,mean_loss,X_test,y_test,1, 64, verbose=True)
            
    replay[state] = reward
    #Q = update_Q(Q, state, 100, 0.01, 1)  
save_obj(replay, 'replay_' + str(np.random.randint(1000)))