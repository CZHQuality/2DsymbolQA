#coding=utf-8
# This code is for testing models trainied by tr3.py

from __future__ import print_function
import os
import matplotlib.pyplot as plt 
import tensorflow as tf 
from PIL import Image
import numpy
import cv2
import random


# Start of First Training Stage
learning_rate = 0.001
batch_size = 100
display_step = 1
SIZE = 64
n_input = SIZE*SIZE
n_classes_1 = 6 # 5 types of Distortion Types
n_classes_2 = 5 # 6 types of Final Quality Grades
n_classes_3 = 2 # 2 types of decoded results: 0 means Fail, 1 means Success
n_classes_4 = 5 # 5 types of 
n_classes_5 = 5 # 5 types of 
n_classes_6 = 5 # 5 types of 
n_classes_7 = 5 # 5 types of 
n_classes_8 = 5 # 5 types of 
n_classes_9 = 5 # 5 types of 

drop_out = 0.75

w_1 = 0.9
w_2 = 0.1
w_3 = 0.1
w_4 = 0.1
w_5 = 0.1
w_6 = 0.1
w_7 = 0.1
w_8 = 0.1
w_9 = 0.1

x = tf.placeholder(tf.float32, [None, SIZE, SIZE, 1]) # input image
y_1 = tf.placeholder(tf.float32, [None, n_classes_1]) # label 1
y_2 = tf.placeholder(tf.float32, [None, n_classes_2]) # label 2
y_3 = tf.placeholder(tf.float32, [None, n_classes_3]) # label 3
y_4 = tf.placeholder(tf.float32, [None, n_classes_4]) # label 4
y_5 = tf.placeholder(tf.float32, [None, n_classes_5]) # label 5
y_6 = tf.placeholder(tf.float32, [None, n_classes_6]) # label 6
y_7 = tf.placeholder(tf.float32, [None, n_classes_7]) # label 7
y_8 = tf.placeholder(tf.float32, [None, n_classes_8]) # label 8
y_9 = tf.placeholder(tf.float32, [None, n_classes_9]) # label 9

keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, SIZE, SIZE, 1])

    #Shared Layers
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)

    conv1 = maxpool2d(conv1, k=2)
    print(conv1.shape)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.shape)

    conv2 = maxpool2d(conv2, k=2)
    print(conv2.shape)

    # Subtask 1 : distortion classification
    conv3_1 = conv2d(conv2, weights['wc3_1'], biases['bc3_1'])
    print(conv3_1.shape)

    conv4_1 = conv2d(conv3_1, weights['wc4_1'], biases['bc4_1'])
    print(conv4_1.shape)

    fc1_1 = tf.reshape(conv4_1, [-1, weights['wd1_1'].get_shape().as_list()[0]])
    fc1_1 = tf.add(tf.matmul(fc1_1, weights['wd1_1']), biases['bd1_1'])
    fc1_1 = tf.nn.relu(fc1_1)
    fc1_1 = tf.nn.dropout(fc1_1, dropout)

    out_1 = tf.add(tf.matmul(fc1_1, weights['out_w_1']), biases['out_b_1'])

    # Subtask 2 : quality grade estimation
    conv3_2 = conv2d(conv2, weights['wc3_2'], biases['bc3_2'])
    print(conv3_2.shape)

    conv4_2 = conv2d(conv3_2, weights['wc4_2'], biases['bc4_2'])
    print(conv4_2.shape)

    fc1_2 = tf.reshape(conv4_2, [-1, weights['wd1_2'].get_shape().as_list()[0]])
    fc1_2 = tf.add(tf.matmul(fc1_2, weights['wd1_2']), biases['bd1_2'])
    fc1_2 = tf.nn.relu(fc1_2)
    fc1_2 = tf.nn.dropout(fc1_2, dropout)

    out_2 = tf.add(tf.matmul(fc1_2, weights['out_w_2']), biases['out_b_2'])

    # Subtask 3 : decode results, 0 means fail, 1 means success
    fc1_3 = tf.reshape(conv4_2, [-1, weights['wd1_3'].get_shape().as_list()[0]])
    fc1_3 = tf.add(tf.matmul(fc1_3, weights['wd1_3']), biases['bd1_3'])
    fc1_3 = tf.nn.relu(fc1_3)
    fc1_3 = tf.nn.dropout(fc1_3, dropout)

    out_3 = tf.add(tf.matmul(fc1_3, weights['out_w_3']), biases['out_b_3'])

    # Subtask 4 : 
    fc1_4 = tf.reshape(conv4_2, [-1, weights['wd1_4'].get_shape().as_list()[0]])
    fc1_4 = tf.add(tf.matmul(fc1_4, weights['wd1_4']), biases['bd1_4'])
    fc1_4 = tf.nn.relu(fc1_4)
    fc1_4 = tf.nn.dropout(fc1_4, dropout)

    out_4 = tf.add(tf.matmul(fc1_4, weights['out_w_4']), biases['out_b_4'])

    # Subtask 5 : 
    fc1_5 = tf.reshape(conv4_2, [-1, weights['wd1_5'].get_shape().as_list()[0]])
    fc1_5 = tf.add(tf.matmul(fc1_5, weights['wd1_5']), biases['bd1_5'])
    fc1_5 = tf.nn.relu(fc1_5)
    fc1_5 = tf.nn.dropout(fc1_5, dropout)

    out_5 = tf.add(tf.matmul(fc1_5, weights['out_w_5']), biases['out_b_5'])

    # Subtask 6
    fc1_6 = tf.reshape(conv4_2, [-1, weights['wd1_6'].get_shape().as_list()[0]])
    fc1_6 = tf.add(tf.matmul(fc1_6, weights['wd1_6']), biases['bd1_6'])
    fc1_6 = tf.nn.relu(fc1_6)
    fc1_6 = tf.nn.dropout(fc1_6, dropout)

    out_6 = tf.add(tf.matmul(fc1_6, weights['out_w_6']), biases['out_b_6'])

    # Subtask 7 
    fc1_7 = tf.reshape(conv4_2, [-1, weights['wd1_7'].get_shape().as_list()[0]])
    fc1_7 = tf.add(tf.matmul(fc1_7, weights['wd1_7']), biases['bd1_7'])
    fc1_7 = tf.nn.relu(fc1_7)
    fc1_7 = tf.nn.dropout(fc1_7, dropout)

    out_7 = tf.add(tf.matmul(fc1_7, weights['out_w_7']), biases['out_b_7'])

    # Subtask 8
    fc1_8 = tf.reshape(conv4_2, [-1, weights['wd1_8'].get_shape().as_list()[0]])
    fc1_8 = tf.add(tf.matmul(fc1_8, weights['wd1_8']), biases['bd1_8'])
    fc1_8 = tf.nn.relu(fc1_8)
    fc1_8 = tf.nn.dropout(fc1_8, dropout)

    out_8 = tf.add(tf.matmul(fc1_8, weights['out_w_8']), biases['out_b_8'])

    # Subtask 9
    fc1_9 = tf.reshape(conv4_2, [-1, weights['wd1_9'].get_shape().as_list()[0]])
    fc1_9 = tf.add(tf.matmul(fc1_9, weights['wd1_9']), biases['bd1_9'])
    fc1_9 = tf.nn.relu(fc1_9)
    fc1_9 = tf.nn.dropout(fc1_9, dropout)

    out_9 = tf.add(tf.matmul(fc1_9, weights['out_w_9']), biases['out_b_9'])

    return out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9

weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,1,8])),
    'wc2': tf.Variable(tf.random_normal([5,5,8,16])),

    'wc3_1': tf.Variable(tf.random_normal([5,5,16,32])),
    'wc4_1': tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1_1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_1': tf.Variable(tf.random_normal([1024, n_classes_1])),

    'wc3_2': tf.Variable(tf.random_normal([5,5,16,32])),
    'wc4_2': tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1_2': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_2': tf.Variable(tf.random_normal([1024, n_classes_2])),

    'wd1_3': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_3': tf.Variable(tf.random_normal([1024, n_classes_3])),

    'wd1_4': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_4': tf.Variable(tf.random_normal([1024, n_classes_4])),

    'wd1_5': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_5': tf.Variable(tf.random_normal([1024, n_classes_5])),

    'wd1_6': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_6': tf.Variable(tf.random_normal([1024, n_classes_6])),

    'wd1_7': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_7': tf.Variable(tf.random_normal([1024, n_classes_7])),

    'wd1_8': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_8': tf.Variable(tf.random_normal([1024, n_classes_8])),

    'wd1_9': tf.Variable(tf.random_normal([16*16*64, 1024])),
    'out_w_9': tf.Variable(tf.random_normal([1024, n_classes_9]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([8])),
    'bc2': tf.Variable(tf.random_normal([16])),

    'bc3_1': tf.Variable(tf.random_normal([32])),
    'bc4_1': tf.Variable(tf.random_normal([64])),
    'bd1_1': tf.Variable(tf.random_normal([1024])),
    'out_b_1': tf.Variable(tf.random_normal([n_classes_1])),

    'bc3_2': tf.Variable(tf.random_normal([32])),
    'bc4_2': tf.Variable(tf.random_normal([64])),
    'bd1_2': tf.Variable(tf.random_normal([1024])),
    'out_b_2': tf.Variable(tf.random_normal([n_classes_2])),

    'bd1_3': tf.Variable(tf.random_normal([1024])),
    'out_b_3': tf.Variable(tf.random_normal([n_classes_3])),

    'bd1_4': tf.Variable(tf.random_normal([1024])),
    'out_b_4': tf.Variable(tf.random_normal([n_classes_4])),

    'bd1_5': tf.Variable(tf.random_normal([1024])),
    'out_b_5': tf.Variable(tf.random_normal([n_classes_5])),

    'bd1_6': tf.Variable(tf.random_normal([1024])),
    'out_b_6': tf.Variable(tf.random_normal([n_classes_6])),

    'bd1_7': tf.Variable(tf.random_normal([1024])),
    'out_b_7': tf.Variable(tf.random_normal([n_classes_7])),

    'bd1_8': tf.Variable(tf.random_normal([1024])),
    'out_b_8': tf.Variable(tf.random_normal([n_classes_8])),

    'bd1_9': tf.Variable(tf.random_normal([1024])),
    'out_b_9': tf.Variable(tf.random_normal([n_classes_9]))
}

[pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9] = conv_net(x, weights, biases, keep_prob) #??
pred_result_1 = tf.argmax(pred_1, 1)
pred_result_2 = tf.argmax(pred_2, 1)
pred_result_3 = tf.argmax(pred_3, 1)
pred_result_4 = tf.argmax(pred_4, 1)
pred_result_5 = tf.argmax(pred_5, 1)
pred_result_6 = tf.argmax(pred_6, 1)
pred_result_7 = tf.argmax(pred_7, 1)
pred_result_8 = tf.argmax(pred_8, 1)
pred_result_9 = tf.argmax(pred_9, 1)


#loss function of subtask 1
cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_1, labels=y_1))
#loss function of subtask 2
cost_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_2, labels=y_2))
#loss function of subtask 3
cost_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_3, labels=y_3))
#loss function of subtask 4
cost_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_4, labels=y_4))
#loss function of subtask 5
cost_5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_5, labels=y_5))
#loss function of subtask 6
cost_6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_6, labels=y_6))
#loss function of subtask 7
cost_7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_7, labels=y_7))
#loss function of subtask 8
cost_8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_8, labels=y_8))
#loss function of subtask 9
cost_9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_9, labels=y_9))

cost = w_1*cost_1 + w_2*cost_2 + w_3*cost_3 + w_4*cost_4 + w_5*cost_5 + w_6*cost_6 + w_7*cost_7 + w_8*cost_8 + w_9*cost_9
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred_1 = tf.equal(tf.argmax(pred_1, 1), tf.argmax(y_1, 1))
correct_pred_2 = tf.equal(tf.argmax(pred_2, 1), tf.argmax(y_2, 1))
correct_pred_3 = tf.equal(tf.argmax(pred_3, 1), tf.argmax(y_3, 1))
correct_pred_4 = tf.equal(tf.argmax(pred_4, 1), tf.argmax(y_4, 1))
correct_pred_5 = tf.equal(tf.argmax(pred_5, 1), tf.argmax(y_5, 1))
correct_pred_6 = tf.equal(tf.argmax(pred_6, 1), tf.argmax(y_6, 1))
correct_pred_7 = tf.equal(tf.argmax(pred_7, 1), tf.argmax(y_7, 1))
correct_pred_8 = tf.equal(tf.argmax(pred_8, 1), tf.argmax(y_8, 1))
correct_pred_9 = tf.equal(tf.argmax(pred_9, 1), tf.argmax(y_9, 1))

accuracy_1 = tf.reduce_mean(tf.cast(correct_pred_1, tf.float32))
accuracy_2 = tf.reduce_mean(tf.cast(correct_pred_2, tf.float32))
accuracy_3 = tf.reduce_mean(tf.cast(correct_pred_3, tf.float32))
accuracy_4 = tf.reduce_mean(tf.cast(correct_pred_4, tf.float32))
accuracy_5 = tf.reduce_mean(tf.cast(correct_pred_5, tf.float32))
accuracy_6 = tf.reduce_mean(tf.cast(correct_pred_6, tf.float32))
accuracy_7 = tf.reduce_mean(tf.cast(correct_pred_7, tf.float32))
accuracy_8 = tf.reduce_mean(tf.cast(correct_pred_8, tf.float32))
accuracy_9 = tf.reduce_mean(tf.cast(correct_pred_9, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

Path_1 = "/media/ubuntu/CZHhy/BarcodeQA/TIEcode/DATAMultiLabel/Aug/"
list_path = "/media/ubuntu/CZHhy/BarcodeQA/TIEcode/DATAMultiLabel/list.txt"

with tf.Session() as sess:
    
    saver.restore(sess, "/media/ubuntu/CZHhy/BarcodeQA/TIEcode/NewModel/ModelPretraind2.ckpt")
    
    step = 1

    linestrlist = []
    fh = open(list_path, 'r')
    for line in fh.readlines():
        # print(line)
        linestr = line.strip()
        # print(linestr)
        linestrlist.append(linestr)
    fh.close()
    list = linestrlist

    
    for batch_id in range(81, 120):#Training Set Maximum is 124
        # for batch_id in range(0, 5):#Training Set
            batch = list[batch_id * batch_size : batch_id * batch_size + batch_size]
            batch_xs = []
            batch_ys_1 = []
            batch_ys_2 = []
            batch_ys_3 = []
            batch_ys_4 = []
            batch_ys_5 = []
            batch_ys_6 = []
            batch_ys_7 = []
            batch_ys_8 = []
            batch_ys_9 = []
            
            for image in batch:
                score_1 = image[2:3] #label 1, Distortion Type
                score_2 = image[0:1] #label 2, Quality Grade
                score_3 = image[4:5] #label 3, 
                score_4 = image[6:7] #label 4, 
                score_5 = image[8:9] #label 5, 
                score_6 = image[10:11] #label 6, 
                score_7 = image[12:13] #label 7, 
                score_8 = image[14:15] #label 8, 
                score_9 = image[16:17] #label 9, 

                # print("Scores are:", score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8, score_9)
                
                img = cv2.imread(Path_1 + image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (SIZE, SIZE))
                img_ndarray = numpy.asarray(img, dtype='float32')
                img_ndarray = numpy.reshape(img_ndarray, [SIZE,SIZE,1])

                batch_x = img_ndarray
                batch_xs.append(batch_x)
                batch_y_1 = numpy.asarray([0,0,0,0,0,0]) # 6 distortion types
                batch_y_2 = numpy.asarray([0,0,0,0,0]) # 5 quality grades
                batch_y_3 = numpy.asarray([0,0])
                batch_y_4 = numpy.asarray([0,0,0,0,0])
                batch_y_5 = numpy.asarray([0,0,0,0,0])
                batch_y_6 = numpy.asarray([0,0,0,0,0])
                batch_y_7 = numpy.asarray([0,0,0,0,0])
                batch_y_8 = numpy.asarray([0,0,0,0,0])
                batch_y_9 = numpy.asarray([0,0,0,0,0])

                batch_y_1[int(score_1)] = 1 # test code should be changed !!!!!!!!
                batch_y_1 = numpy.reshape(batch_y_1, [6, ])
                batch_ys_1.append(batch_y_1)

                batch_y_2[int(score_2)] = 1
                batch_y_2 = numpy.reshape(batch_y_2, [5, ])
                batch_ys_2.append(batch_y_2)

                batch_y_3[int(score_3)] = 1
                batch_y_3 = numpy.reshape(batch_y_3, [2, ])
                batch_ys_3.append(batch_y_3)

                batch_y_4[int(score_4)] = 1
                batch_y_4 = numpy.reshape(batch_y_4, [5, ])
                batch_ys_4.append(batch_y_4)

                batch_y_5[int(score_5)] = 1
                batch_y_5 = numpy.reshape(batch_y_5, [5, ])
                batch_ys_5.append(batch_y_5)

                batch_y_6[int(score_6)] = 1
                batch_y_6 = numpy.reshape(batch_y_6, [5, ])
                batch_ys_6.append(batch_y_6)

                batch_y_7[int(score_7)] = 1
                batch_y_7 = numpy.reshape(batch_y_7, [5, ])
                batch_ys_7.append(batch_y_7)

                batch_y_8[int(score_8)] = 1
                batch_y_8 = numpy.reshape(batch_y_8, [5, ])
                batch_ys_8.append(batch_y_8)

                batch_y_9[int(score_9)] = 1
                batch_y_9 = numpy.reshape(batch_y_9, [5, ])
                batch_ys_9.append(batch_y_9)





            batch_xs = numpy.asarray(batch_xs)
            print(batch_xs.shape)

            batch_ys_1 = numpy.asarray(batch_ys_1)
            batch_ys_1_2 = sess.run(tf.argmax(batch_ys_1, 1))
            print("Labels 1 are:", batch_ys_1_2)

            batch_ys_2 = numpy.asarray(batch_ys_2)
            batch_ys_2_2 = sess.run(tf.argmax(batch_ys_2, 1))
            print("Labels 2 are:", batch_ys_2_2)

            batch_ys_3 = numpy.asarray(batch_ys_3)
            batch_ys_3_2 = sess.run(tf.argmax(batch_ys_3, 1))
            print("Labels 3 are:", batch_ys_3_2)

            batch_ys_4 = numpy.asarray(batch_ys_4)
            batch_ys_4_2 = sess.run(tf.argmax(batch_ys_4, 1))
            print("Labels 4 are:", batch_ys_4_2)

            batch_ys_5 = numpy.asarray(batch_ys_5)
            batch_ys_5_2 = sess.run(tf.argmax(batch_ys_5, 1))
            print("Labels 5 are:", batch_ys_5_2)

            batch_ys_6 = numpy.asarray(batch_ys_6)
            batch_ys_6_2 = sess.run(tf.argmax(batch_ys_6, 1))
            print("Labels 6 are:", batch_ys_6_2)

            batch_ys_7 = numpy.asarray(batch_ys_7)
            batch_ys_7_2 = sess.run(tf.argmax(batch_ys_7, 1))
            print("Labels 7 are:", batch_ys_7_2)

            batch_ys_8 = numpy.asarray(batch_ys_8)
            batch_ys_8_2 = sess.run(tf.argmax(batch_ys_8, 1))
            print("Labels 8 are:", batch_ys_8_2)

            batch_ys_9 = numpy.asarray(batch_ys_9)
            batch_ys_9_2 = sess.run(tf.argmax(batch_ys_9, 1))
            print("Labels 9 are:", batch_ys_9_2)
            
            
            


            pred_result_test_1 = sess.run(pred_result_1, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 1 Are:", pred_result_test_1)

            pred_result_test_2 = sess.run(pred_result_2, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 2 Are:", pred_result_test_2)

            pred_result_test_3 = sess.run(pred_result_3, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 3 Are:", pred_result_test_3)

            pred_result_test_4 = sess.run(pred_result_4, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 4 Are:", pred_result_test_4)

            pred_result_test_5 = sess.run(pred_result_5, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 5 Are:", pred_result_test_5)

            pred_result_test_6 = sess.run(pred_result_6, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 6 Are:", pred_result_test_6)

            pred_result_test_7 = sess.run(pred_result_7, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 7 Are:", pred_result_test_7)

            pred_result_test_8 = sess.run(pred_result_8, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 8 Are:", pred_result_test_8)

            pred_result_test_9 = sess.run(pred_result_9, feed_dict={x: batch_xs, keep_prob: 1.})
            print("Predicted Results 9 Are:", pred_result_test_9)

            loss, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9 = sess.run([cost, accuracy_1, accuracy_2, accuracy_3,
                                                        accuracy_4, accuracy_5, accuracy_6, accuracy_7, accuracy_8, accuracy_9], 
                                                        feed_dict={x: batch_xs, 
                                                        y_1: batch_ys_1, y_2: batch_ys_2, y_3: batch_ys_3, y_4: batch_ys_4,
                                                        y_5: batch_ys_5, y_6: batch_ys_6, y_7: batch_ys_7, y_8: batch_ys_8, y_9: batch_ys_9,
                                                        keep_prob: 1.})
            print("The test accuracy 1 is :", acc1)
            print("The test accuracy 2 is :", acc2)
            print("The test accuracy 3 is :", acc3)
            print("The test accuracy 4 is :", acc4)
            print("The test accuracy 5 is :", acc5)
            print("The test accuracy 6 is :", acc6)
            print("The test accuracy 7 is :", acc7)
            print("The test accuracy 8 is :", acc8)
            print("The test accuracy 9 is :", acc9)
            



