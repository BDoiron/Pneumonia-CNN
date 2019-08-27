import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import glob
from sklearn.utils import shuffle
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Training to be performed on GPU

# Create target classes
label_dict = {
0: 'Healthy',
1: 'Pneumonia',
}

# Define training and evaluation arrays and labels.
train_X = []
train_Y = []

val_X = []
val_Y = []


# Import, reshape and classify images in directory according to their filenames
# Currently bacterial and virual infections are classified in the same class
# Future work will be done to distinguish between the two types


for filename in glob.glob("chest_xray/train/*/*.*"):
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  #Comvert to grayscale
    new_array = cv2.resize(img_array, (112, 112)) #Reshape all images to 112x112
    train_X.append(new_array)
    if "bacteria" in filename or "virus" in filename:
        train_Y.append(0)
    else:
        train_Y.append(1)


# Reshape to TensorFlow tensor shape
train_X = np.array(train_X).reshape(-1, 112, 112, 1)

for filename in glob.glob("chest_xray/test/*/*.*"):
    img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  #Comvert to grayscale
    new_array = cv2.resize(img_array, (112, 112)) #Reshape all images to 112x112
    val_X.append(new_array)
    if "bacteria" in filename or "virus" in filename:
        val_Y.append(0)
    else:
        val_Y.append(1)


# Reshape to TensorFlow tensor shape
val_X=np.array(val_X).reshape(-1, 112, 112, 1)

# Randomize order of the training and eveluation sets as they're read in sequence
train_X, train_Y = shuffle(train_X, train_Y)
val_X, val_Y = shuffle(val_X, val_Y)

# Reformat to categorial labels
train_Y = to_categorical(train_Y)
val_Y = to_categorical(val_Y)


# Specify training parameters
training_iters = 25
learning_rate = 0.001
batch_size = 100

# Data Input (img shape: 112x112)
n_input = 112

# Total classes (0-9 digits)
n_classes = 2

# Define x and y placeholders to allow for operations to be completed
# and build computational graph without feeding in data

x = tf.compat.v1.placeholder("float", [None, 112, 112, 1])
y = tf.compat.v1.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

weights = {
    'wc1': tf.compat.v1.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.compat.v1.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.compat.v1.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc4': tf.compat.v1.get_variable('W3', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc5': tf.compat.v1.get_variable('W4', shape=(3,3,256,512), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.compat.v1.get_variable('W5', shape=(4*4*512,512), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.compat.v1.get_variable('W6', shape=(512,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

biases = {
    'bc1': tf.compat.v1.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.compat.v1.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.compat.v1.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.compat.v1.get_variable('B3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bc5': tf.compat.v1.get_variable('B4', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.compat.v1.get_variable('B5', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.compat.v1.get_variable('B6', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
}

#Define the neural network architecture

def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 56x56 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 28x28 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14x14 matrix.
    conv3 = maxpool2d(conv3, k=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7x7 matrix.
    conv4 = maxpool2d(conv4, k=2)

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4x4 matrix.
    conv5 = maxpool2d(conv5, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    summary_writer = tf.compat.v1.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = np.array(train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))])
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 validation images
        val_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: val_X,y : val_Y})
        train_loss.append(loss)
        val_loss.append(valid_loss)
        train_accuracy.append(acc)
        val_accuracy.append(val_acc)
        print("Validation Accuracy:","{:.5f}".format(val_acc))
    summary_writer.close()

plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()
