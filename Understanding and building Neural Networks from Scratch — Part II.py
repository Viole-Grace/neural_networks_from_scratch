
# coding: utf-8

# In[2]:


#install required libraries
# !pip install tensorflow


# In[3]:


import tensorflow as tf


# In[4]:


import pandas as pd
import numpy as np


# In[70]:


from tensorflow.examples.tutorials import mnist

digits = mnist.input_data.read_data_sets("tmp/data", one_hot=True)


# In[95]:


print "Images for training : {}".format(len(digits.train.images))
print "Images for testing : {}".format(len(digits.test.images))
print "Shape of input data : {}".format(digits.train.images.shape[1])


# In[96]:


digits.train.images[0]


# In[97]:


print "Sample train image format : {}".format(type(digits.train.labels[0]))

x_tr, y_tr = [],[]
x_te, y_te = [],[]

for img in (digits.train.images):
    x_tr.append(img)
for img in digits.test.images:
    x_te.append(img)
for label in digits.train.labels:
    y_tr.append(label)
for label in digits.test.labels:
    y_te.append(label)


# In[109]:


x_tr, x_te = np.array(x_tr), np.array(x_te)
y_tr, y_te = np.array(y_tr), np.array(y_te)

print "Shape {} Type {}".format(y_tr.shape[1], type(x_tr))


# In[110]:


#define parameters

input_layer_nodes = 1024
hidden_layer_nodes, num_classes = [1024, 768, 512], y_tr.shape[1]
batch_size, epochs = 64, 20
ip_data_shape = x_tr.shape[1]

X, y = tf.placeholder('float', [None, ip_data_shape]), tf.placeholder('float')


# In[111]:


#define and make neural network

def neural_network(data):
    
    input_layer = {'w':tf.Variable(tf.random_normal([ip_data_shape, input_layer_nodes])),
                   'b':tf.Variable(tf.random_normal([input_layer_nodes]))}
    
    hlayer_1 = {'w':tf.Variable(tf.random_normal([input_layer_nodes, hidden_layer_nodes[0]])),
                'b':tf.Variable(tf.random_normal([hidden_layer_nodes[0]]))}
    hlayer_2 = {'w':tf.Variable(tf.random_normal([hidden_layer_nodes[0], hidden_layer_nodes[1]])),
                'b':tf.Variable(tf.random_normal([hidden_layer_nodes[1]]))}
    hlayer_3 = {'w':tf.Variable(tf.random_normal([hidden_layer_nodes[1], hidden_layer_nodes[2]])),
                'b':tf.Variable(tf.random_normal([hidden_layer_nodes[2]]))}
    
    output_layer = {'w':tf.Variable(tf.random_normal([hidden_layer_nodes[2], num_classes])),
                    'b':tf.Variable(tf.random_normal([num_classes]))}
    
#     (data * weights) + bias ==> activation : perceptron

    ip_l = tf.add(tf.matmul(data, input_layer['w']), input_layer['b'])
    ip_l = tf.nn.relu(ip_l)
    
    hl_1 = tf.add(tf.matmul(ip_l, hlayer_1['w']), hlayer_1['b'])
    hl_1 = tf.nn.relu(hl_1)
    hl_2 = tf.add(tf.matmul(hl_1, hlayer_2['w']), hlayer_2['b'])
    hl_2 = tf.nn.relu(hl_2)
    hl_3 = tf.add(tf.matmul(hl_2, hlayer_3['w']), hlayer_3['b'])
    hl_3 = tf.nn.relu(hl_3)
    
    op_l = tf.add(tf.matmul(hl_3, output_layer['w']), output_layer['b'])
    
    return op_l


# In[114]:


def train(x, epochs=20, batch_size=64):
    
    pred = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            ep_loss, i = 0,0
            while i<len(x_tr):
                if i+batch_size < len(x_tr):
                    ep_x = np.array(x_tr[i:i+batch_size])
                    ep_y = np.array(y_tr[i:i+batch_size])
                    i += batch_size
                else:
                    ep_x = np.array(x_tr[i:])
                    ep_y = np.array(y_tr[i:])
                    i += len(x_tr)
                _, c = sess.run([optimizer, cost], feed_dict={x:ep_x , y:ep_y})
                ep_loss += c
            print "Epoch {} of {} completed".format(epoch+1, epochs)
            print "Current epoch loss : {}".format(np.log(ep_loss))
        correct = tf.equal(tf.argmax(pred, -1), tf.argmax(y, -1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        try:
            tr_acc = accuracy.eval({x:x_tr , y:y_tr})
            te_acc = accuracy.eval({x:x_te , y:y_te})
            print "Accuracy :\nTraining - {}\nTesting - {}".format(tr_acc, te_acc)
        except:
            print "Unable to fit accuracy, check tf.argmax()"
            pass
        return pred


# In[115]:


pred = train(X,10,32)

