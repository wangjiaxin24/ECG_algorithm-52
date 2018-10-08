
# python心电分类试验算法

## CNN

传统机器学习框架对于人工特征非常依赖，如果算法设计者没有足够经验，很难提取出高质量的特征，这也是传统机器学习框架的局限性。近几年来以卷积神经网络（Convolutional Neural Network，CNN）为代表的深度学习技术蓬勃兴起，其优势在于可以从大数据中自动习得特征而无需人工设计特征，其在多种任务，例如图像分类与定位，语音识别等领域都展现出了十分强大的性能。

### CNN可用的原因

根据大量的文献，我们可以知道，CNN非常适合处理图像，以至于基于传统机器学习的算法完全不能相比。图像这样的数据形式，存在局部与整体的关系，由低层次特征经组合可形成高层次特征，并可以得到不同特征间的空间相关性。仔细想想，我们的ECG信号似乎也存在这样的特性，局部的一些波形与整体结果息息相关，而诊断，实质上就是由一些低层次的，可见的波形变化抽象成一些疾病的概念，即高层次的特征。而波形的空间关系也往往蕴含了丰富的信息。这样看起来，ECG与图像有很大的相似之处，而CNN通过以下几点可有效利用上述特点：

1）局部连接：CNN网络可提取数据局部特征。

2）权值共享：大大降低了网络训练难度，每个卷积层有多个filter,可提取多种特征。

3）池化操作和多层次结构：实现数据的降维，将低层次局部特征组合无为较高层次特征。
由于我们处理的ECG心拍是一维信号，因此需要使用1维CNN处理。


```python
# -*- coding: utf-8 -*-
"""


===================基于1维CNN的ECG分类算法========================


"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sess=tf.InteractiveSession()

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#使用TensorFlow组件完成CNN网络的搭建，与教程中参数略有不同
def CNNnet(inputs,n_class):
    conv1 = tf.layers.conv1d(inputs=inputs, filters=4, kernel_size=31, strides=1, \
                             padding='same', activation = tf.nn.relu)
    avg_pool_1 = tf.layers.average_pooling1d(inputs=conv1, pool_size=5, strides=5, \
                                         padding='same')
    conv2 = tf.layers.conv1d(inputs=avg_pool_1, filters=8, kernel_size=6, strides=1,\
                             padding='same', activation = tf.nn.relu)
    avg_pool_2 = tf.layers.average_pooling1d(inputs=conv2, pool_size=5, strides=5,\
                                         padding='same')
    
    flat = tf.reshape(avg_pool_2, (-1, int(250/5/5*8)))
    
    logits=tf.layers.dense(inputs=flat, units=n_class, activation=None)
    logits=tf.nn.softmax(logits)
    return logits

#随机获取一个batch大小的数据，用于训练
def get_batch(train_x,train_y,batch_size):
    indices=np.random.choice(train_x.shape[0],batch_size,False)
    batch_x=train_x[indices]
    batch_y=train_y[indices]
    return batch_x,batch_y

#设定路径及文件名并载入，这里的心拍在Matlab下截取完成
#详情：https://blog.csdn.net/qq_15746879/article/details/80340671
Path='C:/Users/Administrator.SC-201604221446/Desktop/Jupyterbook/ECG-ML-DL-Algorithm-Python-master/' #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'

print("Loading data and labels...")
tic=time.time()
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

print("Divide training and testing set...")
train_x=Data[Indices[:10000]]
train_y=Label[Indices[:10000]]
test_x=Data[Indices[10000:]]
test_y=Label[Indices[10000:]]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("1D-CNN setup and initialize...")
tic=time.time()
x=tf.placeholder(tf.float32, [None, 250]) #定义placeholder数据入口
x_=tf.reshape(x,[-1,250,1])
y_=tf.placeholder(tf.float32,[None,4])

logits=CNNnet(x_,4)

learning_rate=0.01
batch_size=16
maxiters=15000

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
#这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
tf.global_variables_initializer().run()
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("1D-CNN training and testing...")
tic=time.time()
for i in range(maxiters):
    batch_x,batch_y=get_batch(train_x,train_y,batch_size)
    train_step.run(feed_dict={x:batch_x,y_:batch_y})
    if i%500==0:
        loss=cost.eval(feed_dict={x:train_x,y_:train_y})
        print("Iteration %d/%d:loss %f"%(i,maxiters,loss))

y_pred=logits.eval(feed_dict={x:test_x,y_:test_y})
y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(test_y,axis=1)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))

Acc=np.mean(y_pred==y_true)
Conf_Mat=confusion_matrix(y_true,y_pred) #利用专用函数得到混淆矩阵
Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])



print('\nAccuracy=%.2f%%'%(Acc*100))
print('Accuracy_N=%.2f%%'%(Acc_N*100))
print('Accuracy_V=%.2f%%'%(Acc_V*100))
print('Accuracy_R=%.2f%%'%(Acc_R*100))
print('Accuracy_L=%.2f%%'%(Acc_L*100))
print('\nConfusion Matrix:\n')
print(Conf_Mat)
print("======================================")
```

    Loading data and labels...
    Divide training and testing set...
    Elapsed time is 0.545031 sec.
    ======================================
    1D-CNN setup and initialize...
    Elapsed time is 1.234071 sec.
    ======================================
    1D-CNN training and testing...
    Iteration 0/15000:loss 1.384970
    Iteration 500/15000:loss 0.786377
    Iteration 1000/15000:loss 0.795672
    Iteration 1500/15000:loss 0.783669
    Iteration 2000/15000:loss 0.780045
    Iteration 2500/15000:loss 0.777241
    Iteration 3000/15000:loss 0.780963
    Iteration 3500/15000:loss 0.777798
    Iteration 4000/15000:loss 0.778603
    Iteration 4500/15000:loss 0.779080
    Iteration 5000/15000:loss 0.776163
    Iteration 5500/15000:loss 0.777095
    Iteration 6000/15000:loss 0.790020
    Iteration 6500/15000:loss 0.781351
    Iteration 7000/15000:loss 0.777560
    Iteration 7500/15000:loss 0.772955
    Iteration 8000/15000:loss 0.775444
    Iteration 8500/15000:loss 0.776259
    Iteration 9000/15000:loss 0.776569
    Iteration 9500/15000:loss 0.777501
    Iteration 10000/15000:loss 0.776842
    Iteration 10500/15000:loss 0.777337
    Iteration 11000/15000:loss 0.780104
    Iteration 11500/15000:loss 0.774626
    Iteration 12000/15000:loss 0.774977
    Iteration 12500/15000:loss 0.780180
    Iteration 13000/15000:loss 0.773879
    Iteration 13500/15000:loss 0.773826
    Iteration 14000/15000:loss 0.781972
    Iteration 14500/15000:loss 0.776121
    Elapsed time is 89.087095 sec.
    
    Accuracy=96.62%
    Accuracy_N=99.84%
    Accuracy_V=97.36%
    Accuracy_R=97.85%
    Accuracy_L=91.40%
    
    Confusion Matrix:
    
    [[2529    0    2    2]
     [  14 2432   33   19]
     [   3    7 2415   43]
     [   6  206    3 2286]]
    ======================================
    

## SVM


```python
# -*- coding: utf-8 -*-
"""

===========基于小波系数特征和支持向量机SVM的ECG分类算法=============


"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#设定路径及文件名并载入，这里的特征来自Matlab中提取的小波系数
#https://blog.csdn.net/qq_15746879/article/details/80340910

print("Loading Features and Labels...")
Path='C:/Users/Administrator.SC-201604221446/Desktop/Jupyterbook/ECG-ML-DL-Algorithm-Python-master/' #自定义路径要正确
FeatureFile='Feature_WT_25.mat'
LabelFile='Label.mat'

tic=time.time()
Feature=load_mat(Path+FeatureFile,'Feature')
Feature=Feature.T
Label=load_mat(Path+LabelFile,'Label')
Label=Label.squeeze()
toc=time.time()

print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

#划分数据集并归一化
print("Data normalizing...")
tic=time.time()
train_x,test_x,train_y,test_y=train_test_split(Feature,Label, test_size=0.5, random_state=42)
min_max_scaler=preprocessing.MinMaxScaler() 
train_x=min_max_scaler.fit_transform(train_x)
test_x= min_max_scaler.transform(test_x)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

#模型训练及预测
print("SVM training and testing...")
tic=time.time()
SVM=SVC(kernel='rbf',C=2,gamma=1)
SVM.fit(train_x,train_y)
y_pred=SVM.predict(test_x)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))

#评估结果
Acc=np.mean(y_pred==test_y)
Conf_Mat=confusion_matrix(test_y,y_pred) #利用专用函数得到混淆矩阵
Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])


print('\nAccuracy=%.2f%%'%(Acc*100))
print('Accuracy_N=%.2f%%'%(Acc_N*100))
print('Accuracy_V=%.2f%%'%(Acc_V*100))
print('Accuracy_R=%.2f%%'%(Acc_R*100))
print('Accuracy_L=%.2f%%'%(Acc_L*100))
print('\nConfusion Matrix:\n')
print(Conf_Mat)

print("======================================")
```

    Loading Features and Labels...
    Elapsed time is 0.065004 sec.
    ======================================
    Data normalizing...
    Elapsed time is 0.012001 sec.
    ======================================
    SVM training and testing...
    Elapsed time is 2.488142 sec.
    
    Accuracy=96.78%
    Accuracy_N=99.72%
    Accuracy_V=91.80%
    Accuracy_R=97.88%
    Accuracy_L=97.81%
    
    Confusion Matrix:
    
    [[2489    5    0    2]
     [  28 2330   90   90]
     [  36    9 2447    8]
     [   8   45    1 2412]]
    ======================================
    

## 单向LSTM

1. 在2维图像中，两个坐标往往被称为“空间”坐标，而在1维ECG信号中，唯一的坐标往往是“时间”坐标。
2. 在CNN的应用中，我们忽略两者的不同，仅关注两者相似的地方，即“局部相关性”和“平移不变性”，从而成功迁移CNN。
3. 但“时间”与“空间”概念终究是不同的，1维ECG信号，说到底，还是一个时间序列。时间序列中往往存在着复杂的因果关系，也是隐藏规律的所在。接触过深度学习领域的人都应该知道，处理时间序列，那就是循环神经网络（RNN）的长项了。理论上，RNN应该也可以处理1维ECG信号。
4. 但是，朴素的RNN有一个致命的问题：长程依赖问题。随后提出的RNN变种——长短时记忆网络（LSTM）可以良好的解决这个问题，因此成为了RNN应用的主要选择。

与CNN这种前馈网络不同，对于RNN来说，系统的输出会保留在网络里，和系统下一刻的输入一起共同决定下一刻的输出。这种特性也使得RNN在处理语义方面非常突出，因为此刻的语义往往依赖于复杂的上文内容。

什么是长程依赖问题？？

即最近输入的信息对结果影响是最大的，而最开始输入的信息已经被网络“遗忘”，很难再对结果产生较大影响。而后提出的LSTM引入了“输入门，输出门，遗忘门”的概念，可以解决这个问题，成为了RNN应用的主流。

而RNN（LSTM）的基本应用模式也可以分为以下几种，有“一对多”，“多对一”，两种“多对多”：

对于我们的ECG应用，我们采用最后一种方式，每个数据点是每个时刻的输入，把握每个时刻的网络输出。但是，我们最终输出的应该是目标信号的类别，而不是一个序列，所以得到每个时刻的网络输出组成的序列后，再送入一个全连接层，全连接层的输出就是One-hot（即000100,0010000）类型的类别编码：

<img style="-webkit-user-select: none;" src="https://img-blog.csdn.net/20180610200126508?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE1NzQ2ODc5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70">

<img style="-webkit-user-select: none;cursor: zoom-in;" src="https://img-blog.csdn.net/20180610205437255" width="673" height="326">

RNN系列网络的训练是BP算法的变种BPTT


```python

#该代码中所用LSTM为单隐层，时间步数为250（与心拍长度一致），分4类。与前面的CNN一样，采用Adam方法训练。
#-*- coding: utf-8 -*-
"""

===================基于单向LSTM(ULSTM)的ECG分类算法========================


"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sess=tf.InteractiveSession()

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#使用TensorFlow组件完成ULSTM网络的搭建
def ULSTM(x,n_input,n_hidden,n_steps,n_classes):
   
    x=tf.transpose(x,[1,0,2])    #整理数据，使之符合ULSTM接口要求
    x=tf.reshape(x,[-1,n_input])
    x=tf.split(x,n_steps)
    
    #以下两句调用TF函数，生成一个单隐层的ULSTM
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    outputs,_=tf.contrib.rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    
    #以下部分将ULSTM每一步的输出拼接，形成特征向量
    for i in range(n_steps):
        if i==0:
            fv=outputs[0]
        else:
            fv=tf.concat([fv,outputs[i]],1)
    fvp=tf.reshape(fv,[-1,1,n_steps*n_hidden,1])
    shp=fvp.get_shape()
    flatten_shape=shp[1].value*shp[2].value*shp[3].value
    
    fvp2=tf.reshape(fvp,[-1,flatten_shape])
    
    #构建最后的全连接层
    weights=tf.Variable(tf.random_normal([flatten_shape,n_classes]))
    biases=tf.Variable(tf.random_normal([n_classes]))
            
    return tf.matmul(fvp2,weights)+biases

#随机获取一个batch大小的数据，用于训练
def get_batch(train_x,train_y,batch_size):
    indices=np.random.choice(train_x.shape[0],batch_size,False)
    batch_x=train_x[indices]
    batch_y=train_y[indices]
    return batch_x,batch_y

#设定路径及文件名并载入，这里的心拍在Matlab下截取完成
#详情：https://blog.csdn.net/qq_15746879/article/details/80340671
Path='C:/Users/Administrator.SC-201604221446/Desktop/Jupyterbook/ECG-ML-DL-Algorithm-Python-master/' #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'

print("Loading data and labels...")
tic=time.time()
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

print("Divide training and testing set...")
train_x=Data[Indices[:10000]]
train_y=Label[Indices[:10000]]
test_x=Data[Indices[10000:]]
test_y=Label[Indices[10000:]]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("ULSTM setup and initialize...")

n_input=1
n_hidden=1
n_steps=250
n_classes=4
tic=time.time()
x=tf.placeholder(tf.float32, [None, 250]) #定义placeholder数据入口
x_=tf.reshape(x,[-1,250,1])
y_=tf.placeholder(tf.float32,[None,4])

logits=ULSTM(x_,n_input,n_hidden,n_steps,n_classes)

learning_rate=0.001
batch_size=16
maxiters=10000

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
#这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
tf.global_variables_initializer().run()
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("ULSTM training and testing...")
tic=time.time()
for i in range(maxiters):
    batch_x,batch_y=get_batch(train_x,train_y,batch_size)
    train_step.run(feed_dict={x:batch_x,y_:batch_y})
    if i%500==0:
        loss=cost.eval(feed_dict={x:train_x,y_:train_y})
        print("Iteration %d/%d:loss %f"%(i,maxiters,loss))

y_pred=logits.eval(feed_dict={x:test_x,y_:test_y})
y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(test_y,axis=1)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))

Acc=np.mean(y_pred==y_true)
Conf_Mat=confusion_matrix(y_true,y_pred) #利用专用函数得到混淆矩阵
Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])


print('\nAccuracy=%.2f%%'%(Acc*100))
print('Accuracy_N=%.2f%%'%(Acc_N*100))
print('Accuracy_V=%.2f%%'%(Acc_V*100))
print('Accuracy_R=%.2f%%'%(Acc_R*100))
print('Accuracy_L=%.2f%%'%(Acc_L*100))
print('\nConfusion Matrix:\n')
print(Conf_Mat)
print("======================================")
```

    Loading data and labels...
    Divide training and testing set...
    Elapsed time is 0.450026 sec.
    ======================================
    ULSTM setup and initialize...
    Elapsed time is 79.489547 sec.
    ======================================
    ULSTM training and testing...
    Iteration 0/10000:loss 1.433699
    Iteration 500/10000:loss 0.450065
    Iteration 1000/10000:loss 0.311411
    Iteration 1500/10000:loss 0.260979
    Iteration 2000/10000:loss 0.228665
    Iteration 2500/10000:loss 0.212020
    Iteration 3000/10000:loss 0.195801
    Iteration 3500/10000:loss 0.184223
    Iteration 4000/10000:loss 0.172765
    Iteration 4500/10000:loss 0.164214
    Iteration 5000/10000:loss 0.160734
    Iteration 5500/10000:loss 0.151483
    Iteration 6000/10000:loss 0.146400
    Iteration 6500/10000:loss 0.142513
    Iteration 7000/10000:loss 0.138893
    Iteration 7500/10000:loss 0.134405
    Iteration 8000/10000:loss 0.130373
    Iteration 8500/10000:loss 0.128658
    Iteration 9000/10000:loss 0.124987
    Iteration 9500/10000:loss 0.123847
    Elapsed time is 596.073093 sec.
    
    Accuracy=96.39%
    Accuracy_N=99.92%
    Accuracy_V=90.89%
    Accuracy_R=98.11%
    Accuracy_L=96.78%
    
    Confusion Matrix:
    
    [[2436    0    2    0]
     [  20 2304   47  164]
     [   1   43 2493    4]
     [   6   71    3 2406]]
    ======================================
    

## 双向LSTM


```python
#-*- coding: utf-8 -*-
"""

===================基于双向LSTM(BiLSTM)的ECG分类算法================

"""
#载入所需工具包
import time
import numpy as np
import h5py as hp
import tensorflow as tf
from sklearn.metrics import confusion_matrix

sess=tf.InteractiveSession()

#载入.mat文件的函数,h5py解码并转换为numpy数组
def load_mat(path_data,name_data,dtype='float32'):
    data=hp.File(path_data)
    arrays_d={}
    for k,v in data.items():
        arrays_d[k]=np.array(v)
    dataArr=np.array(arrays_d[name_data],dtype=dtype)
    return dataArr

#使用TensorFlow组件完成BiLSTM网络的搭建
def BiLSTM(x,n_input,n_hidden,n_steps,n_classes):
   
    x=tf.transpose(x,[1,0,2])    #整理数据，使之符合ULSTM接口要求
    x=tf.reshape(x,[-1,n_input])
    x=tf.split(x,n_steps)
    
    #以下两句调用TF函数，生成一个单隐层的BiLSTM
    lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    
    outputs,_,_=tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                        lstm_bw_cell,
                                                        x,dtype=tf.float32)
    
    #以下部分将BiLSTM每一步的输出拼接，形成特征向量
    for i in range(n_steps):
        if i==0:
            fv=outputs[0]
        else:
            fv=tf.concat([fv,outputs[i]],1)
    fvp=tf.reshape(fv,[-1,1,2*n_steps*n_hidden,1])
    shp=fvp.get_shape()
    flatten_shape=shp[1].value*shp[2].value*shp[3].value
    
    fvp2=tf.reshape(fvp,[-1,flatten_shape])
    
    #构建最后的全连接层
    weights=tf.Variable(tf.random_normal([flatten_shape,n_classes]))
    biases=tf.Variable(tf.random_normal([n_classes]))
            
    return tf.matmul(fvp2,weights)+biases

#随机获取一个batch大小的数据，用于训练
def get_batch(train_x,train_y,batch_size):
    indices=np.random.choice(train_x.shape[0],batch_size,False)
    batch_x=train_x[indices]
    batch_y=train_y[indices]
    return batch_x,batch_y

#设定路径及文件名并载入，这里的心拍在Matlab下截取完成
#详情：https://blog.csdn.net/qq_15746879/article/details/80340671
Path='C:/Users/Administrator.SC-201604221446/Desktop/Jupyterbook/ECG-ML-DL-Algorithm-Python-master/'  #自定义路径要正确
DataFile='Data_CNN.mat'
LabelFile='Label_OneHot.mat'

print("Loading data and labels...")
tic=time.time()
Data=load_mat(Path+DataFile,'Data')
Label=load_mat(Path+LabelFile,'Label')
Data=Data.T
Indices=np.arange(Data.shape[0]) #随机打乱索引并切分训练集与测试集
np.random.shuffle(Indices)

print("Divide training and testing set...")
train_x=Data[Indices[:10000]]
train_y=Label[Indices[:10000]]
test_x=Data[Indices[10000:]]
test_y=Label[Indices[10000:]]
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("BiLSTM setup and initialize...")

n_input=1
n_hidden=1
n_steps=250
n_classes=4
tic=time.time()
x=tf.placeholder(tf.float32, [None, 250]) #定义placeholder数据入口
x_=tf.reshape(x,[-1,250,1])
y_=tf.placeholder(tf.float32,[None,4])

logits=BiLSTM(x_,n_input,n_hidden,n_steps,n_classes)

learning_rate=0.001
batch_size=16
maxiters=10000

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
#这里使用了自适应学习率的Adam训练方法，可以认为是SGD的高级演化版本之一
train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
tf.global_variables_initializer().run()
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))
print("======================================")

print("ULSTM training and testing...")
tic=time.time()
for i in range(maxiters):
    batch_x,batch_y=get_batch(train_x,train_y,batch_size)
    train_step.run(feed_dict={x:batch_x,y_:batch_y})
    if i%500==0:
        loss=cost.eval(feed_dict={x:train_x,y_:train_y})
        print("Iteration %d/%d:loss %f"%(i,maxiters,loss))

y_pred=logits.eval(feed_dict={x:test_x,y_:test_y})
y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(test_y,axis=1)
toc=time.time()
print("Elapsed time is %f sec."%(toc-tic))

Acc=np.mean(y_pred==y_true)
Conf_Mat=confusion_matrix(y_true,y_pred) #利用专用函数得到混淆矩阵
Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])


print('\nAccuracy=%.2f%%'%(Acc*100))
print('Accuracy_N=%.2f%%'%(Acc_N*100))
print('Accuracy_V=%.2f%%'%(Acc_V*100))
print('Accuracy_R=%.2f%%'%(Acc_R*100))
print('Accuracy_L=%.2f%%'%(Acc_L*100))
print('\nConfusion Matrix:\n')
print(Conf_Mat)
print("======================================")
```

    Loading data and labels...
    Divide training and testing set...
    Elapsed time is 0.450026 sec.
    ======================================
    BiLSTM setup and initialize...
    Elapsed time is 121.375942 sec.
    ======================================
    ULSTM training and testing...
    Iteration 0/10000:loss 3.646687
    Iteration 500/10000:loss 0.482028
    Iteration 1000/10000:loss 0.260518
    Iteration 1500/10000:loss 0.181141
    Iteration 2000/10000:loss 0.153584
    Iteration 2500/10000:loss 0.135903
    Iteration 3000/10000:loss 0.121719
    Iteration 3500/10000:loss 0.109796
    Iteration 4000/10000:loss 0.102429
    Iteration 4500/10000:loss 0.099351
    Iteration 5000/10000:loss 0.091623
    Iteration 5500/10000:loss 0.090429
    Iteration 6000/10000:loss 0.083646
    Iteration 6500/10000:loss 0.082062
    Iteration 7000/10000:loss 0.079580
    Iteration 7500/10000:loss 0.078761
    Iteration 8000/10000:loss 0.074174
    Iteration 8500/10000:loss 0.072146
    Iteration 9000/10000:loss 0.069599
    Iteration 9500/10000:loss 0.068184
    Elapsed time is 849.787605 sec.
    
    Accuracy=97.94%
    Accuracy_N=99.88%
    Accuracy_V=94.34%
    Accuracy_R=99.41%
    Accuracy_L=98.08%
    
    Confusion Matrix:
    
    [[2457    0    1    2]
     [   8 2316   45   86]
     [   0   11 2519    4]
     [   0   49    0 2502]]
    ======================================
    
