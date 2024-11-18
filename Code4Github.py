import h5py
import math
import keras
import numpy as np
from dgconv2 import DGC
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from keras.layers import Activation,Input,Dense,Lambda,Conv2D,Concatenate,concatenate,BatchNormalization,GlobalAveragePooling2D
iterations=10000
slide_size=5
data=h5py.File('dataset.mat','r')
source='result/'
loss_weights=[1,1,0.1,0.01]
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
tf.config.experimental_run_functions_eagerly(True)
lr=0.001
X_spatial_all=data['X_spatial'][()].transpose(3,2,1,0)
LiDAR_all=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all=np.expand_dims(LiDAR_all,-1)
act_Y_train_all=data['act_Y_train'][()].transpose(1,0)
indexi_all=data['indexi'][()].transpose(1,0)
indexj_all=data['indexj'][()].transpose(1,0)
X_spatial_all=X_spatial_all.astype('float32')
LiDAR_all=LiDAR_all.astype('float32')
act_Y_train_all=act_Y_train_all.astype('int')
indexi_all=indexi_all.astype('float32')
indexj_all=indexj_all.astype('float32')
act_Y_train_all[act_Y_train_all==-1]=0
dim=X_spatial_all.shape[-1]
scaler=MinMaxScaler(feature_range=(-1,1))
X_spatial_all_=X_spatial_all.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1]*X_spatial_all.shape[2]*X_spatial_all.shape[3]])
LiDAR_all_=LiDAR_all.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1]*LiDAR_all.shape[2]*LiDAR_all.shape[3]])
X_spatial_all_=scaler.fit_transform(X_spatial_all_)
LiDAR_all_=scaler.fit_transform(LiDAR_all_)
X_spatial_all=X_spatial_all_.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1],X_spatial_all.shape[2],X_spatial_all.shape[3]])
LiDAR_all=LiDAR_all_.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1],LiDAR_all.shape[2],LiDAR_all.shape[3]])
act_Y_train_all=np.reshape(act_Y_train_all,act_Y_train_all.shape[0])
indexi_all=np.reshape(indexi_all,indexi_all.shape[0])
indexj_all=np.reshape(indexj_all,indexj_all.shape[0])
act_Y_train=act_Y_train_all
randpaixv=act_Y_train_all.argsort()
X_spatial_all=X_spatial_all[randpaixv]
LiDAR_all=LiDAR_all[randpaixv]
indexi_all=indexi_all[randpaixv]
indexj_all=indexj_all[randpaixv]
act_Y_train_all=act_Y_train_all[randpaixv]
X_spatial_all=X_spatial_all[act_Y_train_all>0]
LiDAR_all=LiDAR_all[act_Y_train_all>0]
indexi_all=indexi_all[act_Y_train_all>0]
indexj_all=indexj_all[act_Y_train_all>0]
act_Y_train_all=act_Y_train_all[act_Y_train_all>0]
indices=np.arange(X_spatial_all.shape[0])
indices_train,indices_test,act_Y_train_train,act_Y_train_test=train_test_split(indices,act_Y_train_all,test_size=0.99,stratify=act_Y_train_all)
X_spatial_train=X_spatial_all[indices_train,:,:]
LiDAR_train=LiDAR_all[indices_train,:,:]
act_Y_train_train=act_Y_train_all[indices_train]
indexi_train=indexi_all[indices_train]
indexj_train=indexj_all[indices_train]
X_spatial_test=X_spatial_all[indices_test,:,:]
LiDAR_test=LiDAR_all[indices_test,:,:]
act_Y_train_test=act_Y_train_all[indices_test]
indexi_test=indexi_all[indices_test]
indexj_test=indexj_all[indices_test]
act_Y_train_train=np_utils.to_categorical(act_Y_train_train-1)
act_Y_train_test=np_utils.to_categorical(act_Y_train_test-1)
def divnorm(args):
    return tf.tile(tf.expand_dims(tf.norm(args[0],axis=-1),axis=1),[1,args[1]])
def Litijiao(args):
    a,b,c,d=args
    a=tf.divide(a,divnorm([a,d]))
    b=tf.divide(b,divnorm([b,d]))
    c=tf.divide(c,divnorm([c,d]))
    ab=tf.sqrt(tf.abs(2-2*tf.divide((a*b),(divnorm([a,d])*divnorm([b,d])))))
    ac=tf.sqrt(tf.abs(2-2*tf.divide((a*c),(divnorm([a,d])*divnorm([c,d])))))
    bc=tf.sqrt(tf.abs(2-2*tf.divide((b*c),(divnorm([b,d])*divnorm([c,d])))))
    p=(ab+ac+bc)/2
    sin=tf.divide((ab*ac*bc),(4*tf.sqrt(p*(p-ab)*(p-ac)*(p-bc))))
    cos=tf.sqrt(1-sin*sin)
    return 2*math.pi*(1-cos) 
activation='tanh'
kernel_regularizer=None
H_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))
L_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))
H_single=H_input
L_single=L_input
Hx1=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H_single)
Hx1=BatchNormalization()(Hx1)
Hx1=Activation(activation)(Hx1)
Hx2=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx1)
Hx2=BatchNormalization()(Hx2)
Hx2=Activation(activation)(Hx2)
Hx3=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx2)
Hx3=BatchNormalization()(Hx3)
Hx3=Activation(activation)(Hx3)
Hx4=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Hx3)
Hx4=BatchNormalization()(Hx4)
Hx4=Activation(activation)(Hx4)
Lx1=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L_single)
Lx1=BatchNormalization()(Lx1)
Lx1=Activation(activation)(Lx1)
Lx2=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx1)
Lx2=BatchNormalization()(Lx2)
Lx2=Activation(activation)(Lx2)
Lx3=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx2)
Lx3=BatchNormalization()(Lx3)
Lx3=Activation(activation)(Lx3)
Lx4=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(Lx3)
Lx4=BatchNormalization()(Lx4)
Lx4=Activation(activation)(Lx4)
merge_x1 = concatenate([Hx1,Lx1],axis=3)
merge_x1_1=DGC(rank=2,filters=int(Hx1.shape[-1]),kernel_size=(3,3),edge=int(np.ceil(np.log2(int(merge_x1.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x1)
merge_x1_1=BatchNormalization()(merge_x1_1)
merge_x2 = concatenate([Hx2,merge_x1_1,Lx2],axis=3)
merge_x2_1=DGC(rank=2,filters=int(Hx2.shape[-1]),kernel_size=(3,3),edge=int(np.ceil(np.log2(int(merge_x2.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x2)
merge_x2_1=BatchNormalization()(merge_x2_1)
merge_x3 = concatenate([Hx3,merge_x2_1,Lx3],axis=3)
merge_x3_1=DGC(rank=2,filters=int(Hx3.shape[-1]),kernel_size=(3,3),edge=int(np.ceil(np.log2(int(merge_x3.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x3)
merge_x3_1=BatchNormalization()(merge_x3_1)
merge_x4 = concatenate([Hx4,merge_x3_1,Lx4],axis=3)
merge_x4_1=DGC(rank=2,filters=int(Hx4.shape[-1]),kernel_size=(3,3),edge=int(np.ceil(np.log2(int(merge_x4.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x4)
merge_x4_1=BatchNormalization()(merge_x4_1)
HL=Concatenate()([Hx4,merge_x4_1,Lx4])
HL1=Conv2D(int(dim*2),3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(HL)
HL1=BatchNormalization()(HL1)
HL1=Activation(activation)(HL1)
H_out1=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(HL1)
H_out1=Activation(activation)(H_out1)
H_out2=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H_out1)
H_out2=Activation(activation)(H_out2)
H_out3=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H_out2)
H_out3=Activation(activation)(H_out3)
H_out4=Conv2D(X_spatial_train.shape[3],3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(H_out3)
H_out=Activation(activation)(H_out4)
L_out1=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(HL1)
L_out1=Activation(activation)(L_out1)
L_out2=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L_out1)
L_out2=Activation(activation)(L_out2)
L_out3=Conv2D(dim,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L_out2)
L_out3=Activation(activation)(L_out3)
L_out4=Conv2D(1,3,strides=(1,1),padding='same',kernel_regularizer=kernel_regularizer)(L_out3)
L_out=Activation(activation)(L_out4)
HL1=GlobalAveragePooling2D()(HL1)
H_single=GlobalAveragePooling2D()(H_single)
L_single=GlobalAveragePooling2D()(L_single)
Hx2=GlobalAveragePooling2D()(Hx2)
Hx3=GlobalAveragePooling2D()(Hx3)
Hx4=GlobalAveragePooling2D()(Hx4)
merge_x1_1=GlobalAveragePooling2D()(merge_x1_1)
merge_x2_1=GlobalAveragePooling2D()(merge_x2_1)
merge_x3_1=GlobalAveragePooling2D()(merge_x3_1)
Lx2=GlobalAveragePooling2D()(Lx2)
Lx3=GlobalAveragePooling2D()(Lx3)
Lx4=GlobalAveragePooling2D()(Lx4)
H_out1=GlobalAveragePooling2D()(H_out1)
H_out2=GlobalAveragePooling2D()(H_out2)
H_out3=GlobalAveragePooling2D()(H_out3)
H_out=GlobalAveragePooling2D()(H_out)
L_out1=GlobalAveragePooling2D()(L_out1)
L_out2=GlobalAveragePooling2D()(L_out2)
L_out3=GlobalAveragePooling2D()(L_out3)
L_out=GlobalAveragePooling2D()(L_out)
out=Dense(act_Y_train_train.shape[1],activation='softmax')(HL1)
loss_H=Lambda(lambda x:K.sqrt(K.sum(K.square(x[0]-x[1]),axis=-1)),output_shape=[1,])([H_single,H_out])
loss_L=Lambda(lambda x:K.sqrt(K.sum(K.square(x[0]-x[1]),axis=-1)),output_shape=[1,])([L_single,L_out])
xent_loss=Lambda(lambda x:x[0]+x[1],output_shape=[1,])([loss_H,loss_L])
kl_loss_out1=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1),name='kl_loss1')([H_out1,L_out1])
kl_loss_out2=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1),name='kl_loss2')([H_out2,L_out2])
kl_loss_out3=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1),name='kl_loss3')([H_out3,L_out3])
kl_loss_out4=Lambda(lambda x:-0.5*K.sum(1+x[1]-K.square(x[0])-K.exp(x[1]),axis=-1),name='kl_loss4')([H_out,L_out])
kl_loss_out=Lambda(lambda x:x[0]+x[1]+x[2]+x[3],output_shape=[1,])([kl_loss_out1,kl_loss_out2,kl_loss_out3,kl_loss_out4])
spatial_loss1=Lambda(lambda x:K.sum(Litijiao([x[0],x[1],x[2],dim]),axis=-1),name='spatial_loss1')([Hx2,merge_x1_1,Lx2])
spatial_loss2=Lambda(lambda x:K.sum(Litijiao([x[0],x[1],x[2],dim]),axis=-1),name='spatial_loss2')([Hx3,merge_x2_1,Lx3])
spatial_loss3=Lambda(lambda x:K.sum(Litijiao([x[0],x[1],x[2],dim]),axis=-1),name='spatial_loss3')([Hx4,merge_x3_1,Lx4])
spatial_loss=Lambda(lambda x:x[0]+x[1]+x[2],output_shape=[1,])([spatial_loss1,spatial_loss2,spatial_loss3])
network=keras.models.Model([H_input,L_input],[out,xent_loss,kl_loss_out,spatial_loss])
network.compile(loss=['categorical_crossentropy','mean_squared_error','mean_squared_error','mean_squared_error'],loss_weights=loss_weights,optimizer=keras.optimizers.Adam(lr=lr,decay=0.01))
network.summary()     
history=network.fit([X_spatial_train,LiDAR_train],[act_Y_train_train,np.zeros([act_Y_train_train.shape[0],1]),np.zeros([act_Y_train_train.shape[0],1]),np.zeros([act_Y_train_train.shape[0],1])],batch_size=128,epochs=iterations,shuffle=True,verbose=1)
Test_loss = network.predict([X_spatial_test,LiDAR_test])
predicted = Test_loss[0]
predicted_label=predicted.argmax(axis=1)
raw_label=act_Y_train_test.argmax(axis=1)
acc=accuracy_score(predicted_label,raw_label)