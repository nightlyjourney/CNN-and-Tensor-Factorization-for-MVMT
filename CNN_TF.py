import scipy.io as scio
import tensorflow as tf
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

#for i in range(4)
dataFile = '/home/dinghao/CNN MTMV/fox_data.mat'
data = scio.loadmat(dataFile)
view_index = data['view_index']
train_label = data['train_label']
test_label = data['test_label']
train_data = data['train_data']
test_data = data['test_data']
valid_label = data['valid_label']
valid_data = data['valid_data']

train_instance_num = train_data.shape[0]
valid_instance_num = valid_data.shape[0]
test_instance_num = test_data.shape[0]

epoch = 30
batch_size = 16
lambda_ = 0.0001

width = int(view_index[0,0])
height = int(view_index[0,1])-width
nLabel = train_label.shape[1]

sess = tf.InteractiveSession()

# Placeholders (MNIST image:28x28pixels=784, label=10)
x = tf.placeholder(tf.float32, shape=[None, width*height]) # [None, 996*996]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 4]

#mini-batch   
def get_next_batch(train_data, train_label, batch_index):
   batch = [train_data[batch_index,:], train_label[batch_index,:]]
   return batch
   
## Weight Initialization    
def weight_variable(name, W_shape,lambda_):
#  initial = tf.truncated_normal(shape, stddev=0.01)
  initial = tf.get_variable(name, shape=W_shape,
           initializer=tf.contrib.layers.xavier_initializer())
  tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda_)(initial))
  return initial

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_4x4(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

# First Convolutional Layer
W_conv1 = weight_variable('W_conv1',[25, 25, 1, 32],lambda_)  # shape of weight tensor = [10,10,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
input_x = tf.reshape(x, [-1,width,height,1]) #
print(input_x.get_shape) # (?, 996, 996, 1)  # -> output image: 28x28 x1

# input_x * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv2d(input_x, W_conv1) + b_conv1)
print(h_conv1.get_shape) # (?, 996, 996, 1)  # -> output image: 28x28 x32
h_pool1 = max_pool_4x4(h_conv1)
print(h_pool1.get_shape) # (?, 249, 249, 1)  # -> output image: 14x14 x32

# Second Convolutional Layer
#W_conv2 = weight_variable('W_conv2',[15, 15, 32, 32],lambda_) # [5, 5, 32, 64]
#b_conv2 = bias_variable([32]) # [64]

#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#print(h_conv2.get_shape) # (?, 249, 249, 64)  # -> output image: 14x14 x64
#h_pool2 = max_pool_4x4(h_conv2)
#print(h_pool2.get_shape) # (?, 63, 63, 32)    # -> output image: 7x7 x64

## Densely Connected Layer (or fully-connected layer)
#W_fc1 = weight_variable('W_fc1',[63*63*32, 256],lambda_)
#b_fc1 = bias_variable([256])

h_pool1_flat = tf.reshape(h_pool1, [-1, 249*249*32])  # -> output image: [-1, 7*7*64] = 3136
print(h_pool1_flat.get_shape)
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
#print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024

# Dropout (to reduce overfitting; useful when training very large neural network)
# Turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_pool1, keep_prob)
#print(h_fc1_drop.get_shape)  # -> output: 1024

# Readout Layer
W_fc2 = weight_variable('W_fc2',[249*249*32, nLabel],lambda_) # [128, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.nn.softmax(tf.matmul(h_pool1_flat, W_fc2) + b_fc2)
print(y_conv.get_shape)  # -> output: 10

#X*Ut
#W = weight_variable('W',[width+height,4*batch_size],lambda_)

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv+1e-10))
tf.add_to_collection('losses',cross_entropy)
loss = tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)  # 1e-4b 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))#the max index of each row
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
i = 1
start = 0
end = batch_size

train_data = np.concatenate((train_data,valid_data),axis=0)
train_label = np.concatenate((train_label,valid_label),axis=0)
train_instance_num +=valid_instance_num

batch_index = np.array(range(train_instance_num))
np.random.shuffle(batch_index)
while i <= epoch:
    batch = get_next_batch(train_data, train_label, batch_index[start:end])
    batch_num = start/batch_size
    if batch_num%10 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x: valid_data, y_: valid_label, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("epoch %d, batch %d, training accuracy %g"%(i, batch_num, train_accuracy))
        train_loss = loss.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})        
        print("training loss %g"%(train_loss))        
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    start += batch_size
    end = start + batch_size
    if end > train_instance_num:
        np.random.shuffle(batch_index)
        start = 0
        end = batch_size
        i += 1
        
# Evaulate our accuracy on the test data
test_acc_list = []
batch_index = np.array(range(test_instance_num))
index = 0
while index < test_instance_num:
    batch = get_next_batch(test_data, test_label, batch_index[index:index+batch_size])
    acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    test_acc_list.append(acc)
    index += batch_size
test_acc = float(sum(test_acc_list))/len(test_acc_list)
print("test accuracy %g"%test_acc)


