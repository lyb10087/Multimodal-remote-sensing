# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:46:19 2018

@author: danfeng
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio 

from tf_utils import random_mini_batches_standard, convert_to_one_hot
from tensorflow.python.framework import ops
from tfdeterminism import patch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
patch()

import cv2
from sklearn import preprocessing

default_max_hw=6

def mirror_concatenate(x, max_hw=default_max_hw):
    x_extension = cv2.copyMakeBorder(x, 0, max_hw, 0, max_hw, cv2.BORDER_REFLECT)

    return x_extension
# 将输入的图像x进行镜像扩展:在图像四周分别添加对应的像素值，使得图像可以进行卷积等操作而不会丢失边缘信息。
# 函数通过cv2.copyMakeBorder()函数在图像四周分别添加max_hw个像素，添加的像素按照BORDER_REFLECT模式进行填充，


def generate_batch(idx, X, Y, batch_size, ws,row,col, shuffle=False):
    num = len(idx)
    if shuffle:
        np.random.shuffle(idx)

    for i in range(0, num, batch_size):
        bi = np.array(idx)[np.arange(i, min(num, i + batch_size))]
        index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)
        index_col = (bi + 1) - (index_row - 1) * col
        # index_row += hw - 1
        # index_col += hw - 1
        patches = np.zeros([bi.size, ws*ws*X.shape[-1]]) 


        for j in range(bi.size):
            a = index_row[j] - 1
            b = index_col[j] - 1
            # 访问数组元素，索引从0开始
            patch = X[a:a + ws, b:b + ws, :]
            patches[j, :] = patch.reshape(ws*ws*X.shape[-1])
            #取出patches的第j行的所有列，将patch放到patches第j行里
        labels = Y[bi]
        # Y[bi]返回的是Y中索引为bi的那些元素构成的数组，即当前batch中样本对应的标签数组
        labels[labels==0]=1
        labels = convert_to_one_hot(labels-1, 15)
        labels = labels.T

        yield patches,labels

def generate_cube(idx, X, Y, ws,row,col, shuffle=False):
    num = len(idx)   #2832
    if shuffle:
        np.random.shuffle(idx)

    bi = np.array(idx) # 重新创建一个bi，这样操作bi不会影响到idx
    index_row = np.ceil((bi + 1)/ col).astype(np.int32) # bi+1相当于对bi里的每一个数+1（np是从0开始索引的）
    index_col = (bi + 1) - (index_row - 1) * col # 数组是从0开始的，0号所以实际是第一个位置。所以bi+1
    patches = np.zeros([bi.size, ws*ws*X.shape[-1]]) # bi.size是行数，ws*ws*X.shape[-1]=7*7*144
    # 每个批次对应一个patch,总批次个的7*7*144=7056，一行相当于一个patch的像素特征
    for j in range(bi.size):
        a = index_row[j] - 1  # 数组索引从0开始
        b = index_col[j] - 1
        patch = X[a:a + ws, b:b+ws, :]
        patches[j, :] = patch.reshape(ws*ws*X.shape[-1])
    labels = Y[bi]-1
    # 转为one-hot之前需要将label都-1才能使用
    labels = convert_to_one_hot(labels, 15)
    labels = labels.T

    return patches,labels

def sampling(Y_train,Y_test):
    n_class = Y_test.max() # Y_test是一个一维数组，所以可以索引出最大值。
    train_idx = list()
    test_idx = list()

    for i in range(1, n_class + 1):
        train_i = np.where(Y_train == i)[0] # 获取索引下标
        test_i = np.where(Y_test == i)[0]

        train_idx.extend(train_i) # 将数组中的元素全部添加进list
        test_idx.extend(test_i)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return train_idx, test_idx


def create_placeholders(n_x1, n_y):
   
    isTraining = tf.placeholder_with_default(True, shape=())
    x1 = tf.placeholder(tf.float32, [None, n_x1], name = "x1")
    y = tf.placeholder(tf.float32, [None, n_y], name = "Y")
    
    return x1, y, isTraining

def initialize_parameters():

    
    tf.set_random_seed(1)
     
    x1_conv_w1 = tf.get_variable("x1_conv_w1", [3,3,144,16], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b1 = tf.get_variable("x1_conv_b1", [16], initializer = tf.zeros_initializer())
  
    x1_conv_w2 = tf.get_variable("x1_conv_w2", [1,1,16,32], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b2 = tf.get_variable("x1_conv_b2", [32], initializer = tf.zeros_initializer())
  
    x1_conv_w3 = tf.get_variable("x1_conv_w3", [3,3,32,64], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b3 = tf.get_variable("x1_conv_b3", [64], initializer = tf.zeros_initializer())

    x1_conv_w4 = tf.get_variable("x1_conv_w4", [1,1,64,128], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b4 = tf.get_variable("x1_conv_b4", [128], initializer = tf.zeros_initializer())

    x1_conv_w5 = tf.get_variable("x1_conv_w5", [1,1,128,15], dtype=tf.float32, initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b5 = tf.get_variable("x1_conv_b5", [15], initializer = tf.zeros_initializer())     
    
    parameters = {"x1_conv_w1": x1_conv_w1,
                  "x1_conv_b1": x1_conv_b1,
                  "x1_conv_w2": x1_conv_w2,
                  "x1_conv_b2": x1_conv_b2,
                  "x1_conv_w3": x1_conv_w3,
                  "x1_conv_b3": x1_conv_b3,
                  "x1_conv_w4": x1_conv_w4,
                  "x1_conv_b4": x1_conv_b4,
                  "x1_conv_w5": x1_conv_w5,
                  "x1_conv_b5": x1_conv_b5}
    
    return parameters


def mynetwork(x1, parameters, isTraining):

    
    x1 = tf.reshape(x1, [-1, 7, 7, 144], name = "x1")
    
    with tf.name_scope("encoder_layer_1"):
         
         x1_conv_layer_z1 = tf.nn.conv2d(x1, parameters['x1_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b1']                                  
         x1_conv_layer_z1_bn = tf.layers.batch_normalization(x1_conv_layer_z1, momentum = 0.9, training = isTraining)  
         x1_conv_layer_a1 = tf.nn.relu(x1_conv_layer_z1_bn)
                  
    with tf.name_scope("encoder_layer_2"):
         
         x1_conv_layer_z2 = tf.nn.conv2d(x1_conv_layer_a1, parameters['x1_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b2']   
         x1_conv_layer_z2_bn = tf.layers.batch_normalization(x1_conv_layer_z2, momentum = 0.9, training = isTraining)                                             
         x1_conv_layer_z2_po = tf.layers.max_pooling2d(x1_conv_layer_z2_bn, 2, 2, padding='SAME')
         x1_conv_layer_a2 = tf.nn.relu(x1_conv_layer_z2_po)
                  
    with tf.name_scope("encoder_layer_3"):
         
         x1_conv_layer_z3 = tf.nn.conv2d(x1_conv_layer_a2, parameters['x1_conv_w3'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b3']   
         x1_conv_layer_z3_bn = tf.layers.batch_normalization(x1_conv_layer_z3, momentum = 0.9, training = isTraining)                                             
         x1_conv_layer_a3 = tf.nn.relu(x1_conv_layer_z3_bn)

    with tf.name_scope("encoder_layer_4"):
         
         x1_conv_layer_z4 = tf.nn.conv2d(x1_conv_layer_a3, parameters['x1_conv_w4'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b4']   
         x1_conv_layer_z4_bn = tf.layers.batch_normalization(x1_conv_layer_z4, momentum = 0.9, training = isTraining)                                             
         x1_conv_layer_z4_po = tf.layers.max_pooling2d(x1_conv_layer_z4_bn, 2, 2, padding='SAME')
         x1_conv_layer_a4 = tf.nn.relu(x1_conv_layer_z4_po)     
         
         x1_conv_layer_a4_po = tf.layers.average_pooling2d(x1_conv_layer_a4, 2, 2, padding='SAME')
          
    with tf.name_scope("encoder_layer_5"):
         
         x1_conv_layer_z5 = tf.nn.conv2d(x1_conv_layer_a4_po, parameters['x1_conv_w5'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b5']   
         x1_conv_layer_z5_shape = x1_conv_layer_z5.get_shape().as_list()
         x1_conv_layer_z5_2d = tf.reshape(x1_conv_layer_z5, [-1, x1_conv_layer_z5_shape[1] * x1_conv_layer_z5_shape[2] * x1_conv_layer_z5_shape[3]])

           
    l2_loss =  tf.nn.l2_loss(parameters['x1_conv_w1']) + tf.nn.l2_loss(parameters['x1_conv_w2']) + tf.nn.l2_loss(parameters['x1_conv_w3']) + tf.nn.l2_loss(parameters['x1_conv_w4'])\
               + tf.nn.l2_loss(parameters['x1_conv_w5'])
               
    return x1_conv_layer_z5_2d, l2_loss

def mynetwork_optimaization(y_es, y_re, l2_loss, reg, learning_rate, global_step):
    
    with tf.name_scope("cost"):
        
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_es, labels = y_re)) + reg * l2_loss
    
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def train_mynetwork(x1_train_set, x1_test_set, y_train_set, y_test_set,drawmap_loder,
           learning_rate_base = 0.001, beta_reg = 0.001, num_epochs = 1, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()                       
    tf.set_random_seed(1)                          
    seed = 1                                     
    (m, n_x1) = x1_train_set.shape                        
    (m, n_y) = y_train_set.shape                            

    costs = []                                   
    costs_dev = []
    train_acc = []
    val_acc = []
    correct_prediction = 0
    
    # Create Placeholders of shape (n_x, n_y)
    x1, y, isTraining = create_placeholders(n_x1, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    with tf.name_scope("network"):

         joint_layer, l2_loss = mynetwork(x1, parameters, isTraining)
         
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 30 * m/minibatch_size, 0.5, staircase = True)
    
    with tf.name_scope("optimization"):
         # network optimization
         cost, optimizer = mynetwork_optimaization(joint_layer, y, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         # Calculate the correct predictions
         joint_layerT = tf.transpose(joint_layer)
         yT = tf.transpose(y)
         correct_prediction = tf.equal(tf.argmax(joint_layerT), tf.argmax(yT))
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs + 1):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_standard(x1_train_set, y_train_set, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (batch_x1, batch_y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x1: batch_x1, y: batch_y, isTraining: True})

                epoch_cost += minibatch_cost / (num_minibatches+ 1)
                epoch_acc += minibatch_acc / (num_minibatches + 1)
 
            feature, epoch_cost_dev, epoch_acc_dev = sess.run([joint_layerT, cost, accuracy], feed_dict={x1: x1_test_set, y: y_test_set, isTraining: False})
            # feature (15, 12197)
           
            # Print the cost every epoch
            if print_cost == True and epoch % 50 == 0:
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost, epoch_cost_dev, epoch_acc, epoch_acc_dev))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                train_acc.append(epoch_acc)
                costs_dev.append(epoch_cost_dev)
                val_acc.append(epoch_acc_dev)
        print('drawimg map')
        pred_test=[]
        for data,label in drawmap_loder:
            feature, epoch_cost_dev, epoch_acc_dev = sess.run([joint_layerT, cost, accuracy],
                                                              feed_dict={x1: data, y: label,
                                                                         isTraining: False})
            pred_test.extend(np.array(feature.argmax(axis=0)))

        pred_test = np.array(pred_test)

        # plot the cost      
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plot the accuracy 
        plt.plot(np.squeeze(train_acc))
        plt.plot(np.squeeze(val_acc))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show() 
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
     
        print("save model")
        save_path = saver.save(sess,"./model/model_hsi.ckpt")
        print("save model:{0} Finished".format(save_path))

        return parameters, val_acc, feature,pred_test


# HSI_TrSet = scio.loadmat('HSI_LiDAR_CNN/HSI_TrSet.mat')
# HSI_TeSet = scio.loadmat('HSI_LiDAR_CNN/HSI_TeSet.mat')
#
# TrLabel = scio.loadmat('HSI_LiDAR_CNN/TrLabel.mat')
# TeLabel = scio.loadmat('HSI_LiDAR_CNN/TeLabel.mat')

# HSI_TrSet = HSI_TrSet['HSI_TrSet']
# HSI_TeSet = HSI_TeSet['HSI_TeSet']
#
# TrLabel = TrLabel['TrLabel']
# TeLabel = TeLabel['TeLabel']

# Y_train = convert_to_one_hot(TrLabel-1, 15)
# Y_test = convert_to_one_hot(TeLabel-1, 15)
#
# Y_train = Y_train.T
# Y_test = Y_test.T


'''
predict_map
'''

HSI_MapSet = scio.loadmat('likyou_data/HSI.mat')
ground_truth=scio.loadmat('likyou_data/gt.mat')
TrLabel = scio.loadmat('likyou_data/TRLabel.mat')
TeLabel = scio.loadmat('likyou_data/TSLabel.mat')

HSI_MapSet=HSI_MapSet['HSI']
#print(HSI_MapSet.shape) (349, 1905, 144)
ground_truth=ground_truth['gt']
TrLabel=TrLabel['TRLabel'] # (349, 1905)

TeLabel=TeLabel['TSLabel'] #(349, 1905)

[row, col, n_feature] = HSI_MapSet.shape


HSI_MapSet = HSI_MapSet.reshape(row * col, n_feature)
HSI_MapSet = np.asarray(HSI_MapSet, dtype=np.float32)# 原来是uint16要变成float32
HSI_MapSet=(HSI_MapSet-np.min(HSI_MapSet))/(np.max(HSI_MapSet)-np.min(HSI_MapSet))
# 将HSI_MapSet进行了归一化，使其数值范围在0到1之间。归一化可以将不同特征的取值范围变得一致，有利于后续的处理和分析


HSI_MapSet = HSI_MapSet.reshape(row, col, n_feature)

HSI_MapSet = mirror_concatenate(HSI_MapSet)

TrLabel=TrLabel.reshape(row*col)
TeLabel=TeLabel.reshape(row*col)

ws=7

#print(HSI_MapSet.shape)   (355, 1911, 144)


train_idx, test_idx=sampling(TrLabel,TeLabel)# print(train_idx)    #2832
HSI_TrSet,Y_train=generate_cube(train_idx, HSI_MapSet, TrLabel.reshape(row * col), ws,row,col,shuffle=False)


scio.savemat('likyou_data/HSI_TrSet_reshaped.mat', {'Data': HSI_TrSet})
scio.savemat('likyou_data/Y_train_reshaped.mat', {'Data': Y_train})

HSI_TeSet,Y_test=generate_cube(test_idx, HSI_MapSet, TeLabel.reshape(row * col), ws,row,col,shuffle=False)
scio.savemat('likyou_data/HSI_TeSet_reshaped.mat', {'Data': HSI_TeSet})
scio.savemat('likyou_data/Y_test_reshaped.mat', {'Data': Y_test})

#画图索引

drawall_idx = np.array([j for j, x in enumerate(ground_truth.reshape(row * col).ravel().tolist())])
drawmap_loder = generate_batch(drawall_idx, HSI_MapSet, ground_truth.reshape(row * col), 64, ws,row,col,shuffle=False)



parameters, val_acc, feature,pred_map = train_mynetwork(HSI_TrSet, HSI_TeSet, Y_train, Y_test,drawmap_loder)

scio.savemat('feature.mat', {'feature': feature})
print ("Test Accuracy: %f" % (max(val_acc)))

map=np.zeros(row*col)  # 349*1905个像素块
map[:]=pred_map[:]
# 15个类别每种用一个颜色代表
colormap=np.array([[0, 205, 0],
                    [127, 255, 0],
                    [46, 139, 87],
                    [0, 139, 0],
                    [160, 82, 45],
                    [0, 255, 255],
                    [255, 255, 255],
                    [216, 191, 216],
                    [255, 0, 0],
                    [139, 0, 0],
                    [205, 205, 0],
                    [255, 255, 0],
                    [238, 154, 0],
                    [85, 26, 139],
                    [255, 127, 80]])
num_class = 15
colormap = colormap * 1.0 / 255
X_result = np.zeros((row * col, 3)) # 创建一个RGB的houston图像

for i in range(0, num_class):
    index = np.where(map == i)[0]  # 取出所有值等于i的元素在X_result数组中的下标
    X_result[index, 0] = colormap[i, 0]  # 将这些元素的第0列赋值为colormap中第i个元素的第0列 R
    X_result[index, 1] = colormap[i, 1]  # 将这些元素的第1列赋值为colormap中第i个元素的第1列 G
    X_result[index, 2] = colormap[i, 2]  # 将这些元素的第2列赋值为colormap中第i个元素的第2列 B

X_result = np.reshape(X_result, (row, col, 3))
plt.imsave('map.png',X_result)




HSI_TrSet = scio.loadmat('likyou_data/HSI_TrSet_reshaped.mat')
HSI_TeSet = scio.loadmat('likyou_data/HSI_TeSet_reshaped.mat')
# 这两个变量分别代表训练集和测试集，文件存储了用于模型训练和评估的数据。

TrLabel = scio.loadmat('likyou_data/Y_train_reshaped.mat')
TeLabel = scio.loadmat('likyou_data/Y_test_reshaped.mat')
# 这两个变量代表了训练标签和测试标签。

HSI_TrSet = HSI_TrSet['Data']
HSI_TeSet = HSI_TeSet['Data']

TrLabel = TrLabel['Data']
TeLabel = TeLabel['Data']
