# Kibrom berihu Girum
# 15/02/2019
# U-net for sgementation of cells
# MODEL UNET 
# Import libraries to use 
import os
import sys
import numpy as np 
from keras.models import Model 
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Dense, Flatten, Dropout, Add, BatchNormalization, GlobalAveragePooling3D, Reshape, multiply
from keras.optimizers import Adam, SGD 
from keras import backend as K
from keras.utils.vis_utils import plot_model
import pydot
import graphviz
# Define the parameters of the data to process, change according to your data_size 
img_cols = 100
img_rows = 100
img_depth = 50
img_channels = 1

def se_block(input, filters, ratio=16):
	''' Create a squeeze-excite block
	Args:
	input: input tensor
	filters: number of output filters
	k: width factor
	Returns: a keras tensor
	'''
	init = input
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# filters = init._keras_shape[channel_axis]
	se_shape = (1, 1, filters)
	se = GlobalAveragePooling3D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation='elu', kernel_initializer='he_normal', use_bias=False)(se)
	se = Dense(1, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
	if K.image_data_format() == 'channels_first':
		se = Permute((3, 1, 2))(se)
	init = Conv3D(1, (1, 1, 1), padding='same')(init) # change of size 
	x = multiply([init, se])
	return x

def dice_coef(y_true, y_pred, smooth=1.):
	'''
	y_ture: target 
	y_pred: predicted image from model
	dice_coef =  2*(X n Y) / ( sum(X) + sum(Y))
	'''
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f*y_pred_f)
	return (2.*intersection + smooth ) /  (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	'''
	Compute the cost funciton (loss function) 
	 cost = 1-dice_coef 
	'''
	return 1-dice_coef(y_true, y_pred)


# def a function that performs a convolution 
def convolve_fnc(input_image, num_filters, kernel_size):
	'''
	Input: image, number of filters to use, kernel size(filter size), activation functions and type of padding
	Output: Convolved features
	c= Conv2D(num_filters, kernel_size=(3,3), activation ='relu', padding='same')(inputs)
	'''
	conv_feature1 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(input_image)
	conv_feature2 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')(conv_feature1)
	return conv_feature2


def  midl2(num_filters=16): #0.88mm
	inputs = Input((img_rows, img_cols, img_depth, img_channels), name ='input')

	#First convolution and pooling
	c1 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(inputs)
	c1 = se_block(c1, filters=num_filters, ratio=8)
	pool1 = MaxPooling3D(pool_size=(2,2,2))(c1)

	#Second convolution and pooling
	c2 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool1)
	c2 = se_block(c2, filters=num_filters, ratio=8)
	pool2 = MaxPooling3D(pool_size=(2,2,2))(c2)

	#Fourth 3 x convolutions
	c4 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool2)

	#c5 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c4)
	c6 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c4)
	d2=Dropout(0.2)(c6)

	#Fifth fully connected layers as 1x1x1 convolution
	fc1 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(c6)
	#fc2 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(fc1)
	#fc2 = Conv3D(21, kernel_size=(1,1,1), activation='elu', padding='same')(fc2)
	fc3 = Conv3D(21, kernel_size=(1,1,1), activation='elu', padding='same')(fc1)
	d2=Dropout(0.2)(fc3)
	# Output layers
	f1=Flatten()(d2)
	#output1=Dense(21, activation='linear')(f1)
	output2=Dense(21, activation='linear',name='output1')(f1)

	model = Model(inputs =[inputs], outputs=[output2])
	model.compile(optimizer = Adam(lr=5e-4), loss = {'output1':'mean_squared_logarithmic_error'}, metrics = {'output1':'mean_absolute_error'}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef SGD(lr=0.01)

	return model


def  midl1(num_filters=16):
	inputs = Input((img_rows, img_cols, img_depth, img_channels), name ='input')

	#First convolution and pooling
	c1 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(inputs)
	c1 = se_block(c1, filters=num_filters, ratio=8)
	pool1 = MaxPooling3D(pool_size=(2,2,2))(c1)

	#Second convolution and pooling
	c2 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool1)
	c2 = se_block(c2, filters=num_filters, ratio=8)
	pool2 = MaxPooling3D(pool_size=(2,2,2))(c2)

	#Fourth 3 x convolutions
	c4 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool2)
	#c5 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c4)
	c6 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c4)
	d2=Dropout(0.2)(c6)

	#Fifth fully connected layers as 1x1x1 convolution
	fc1 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(c6)
	#fc2 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(fc1)
	#fc2 = Conv3D(21, kernel_size=(1,1,1), activation='elu', padding='same')(fc2)
	fc3 = Conv3D(21, kernel_size=(1,1,1), activation='elu', padding='same')(fc1)
	d2=Dropout(0.2)(fc3)
	# Output layers
	f1=Flatten()(d2)
	#output1=Dense(21, activation='linear')(f1)
	output2=Dense(21, activation='linear',name='output1')(f1)

	model = Model(inputs =[inputs], outputs=[output2])
	model.compile(optimizer = Adam(lr=5e-4), loss = {'output1':'mean_squared_logarithmic_error'}, metrics = {'output1':'mean_absolute_error'}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef SGD(lr=0.01)

	return model

def  midl(num_filters=16): #surgetica original
	inputs = Input((img_rows, img_cols, img_depth, img_channels), name ='input1')

	#First convolution and pooling
	c1 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(inputs)
	c1 = se_block(c1, filters=num_filters, ratio=8)
	pool1 = MaxPooling3D(pool_size=(2,2,2))(c1)

	#Second convolution and pooling
	c2 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool1)
	c2 = se_block(c2, filters=num_filters, ratio=8)
	pool2 = MaxPooling3D(pool_size=(2,2,2))(c2)

	#Second convolution and pooling
	c3 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool2)
	c3 = se_block(c3, filters=num_filters, ratio=8)
	pool2 = MaxPooling3D(pool_size=(2,2,2))(c3)

	#Fourth 3 x convolutions
	c4 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool2)
	c5 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c4)
	c6 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c5)

	#Fifth fully connected layers as 1x1x1 convolution
	fc1 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(c6)
	fc2 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(fc1)
	fc3 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(fc2)
	d2=Dropout(0.2)(fc3)
	# Output layers
	f1=Flatten()(d2)
	#output1=Dense(21, activation='linear')(f1)
	output2=Dense(21, activation='linear',name='output1')(f1)

	model = Model(inputs =[inputs], outputs=[output2])
	model.compile(optimizer = Adam(lr=5e-4), loss = {'output1':'mean_squared_logarithmic_error'}, metrics = {'output1':'mean_absolute_error'}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef SGD(lr=0.01)
	return model

def  midlShape(num_filters=16): #surgetica original
	input1 = Input((img_rows, img_cols, img_depth, img_channels), name ='input1')
	input2 = Input((21,1), name ='input2')
	input3 = Reshape((21,))(input2)

	#First convolution and pooling
	c1 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(input1)
	c1 = se_block(c1, filters=num_filters, ratio=8)
	pool1 = MaxPooling3D(pool_size=(2,2,2))(c1)

	#Second convolution and pooling
	c2 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool1)
	c2 = se_block(c2, filters=num_filters, ratio=8)
	pool2 = MaxPooling3D(pool_size=(2,2,2))(c2)

	#Second convolution and pooling
	c3 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool2)
	c3 = se_block(c3, filters=num_filters, ratio=8)
	pool2 = MaxPooling3D(pool_size=(2,2,2))(c3)

	#Fourth 3 x convolutions
	c4 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(pool2)
	c5 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c4)
	c6 = Conv3D(num_filters, kernel_size=(3,3,3), activation='elu', padding='same')(c5)

	#Fifth fully connected layers as 1x1x1 convolution
	fc1 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(c6)
	fc2 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(fc1)
	fc3 = Conv3D(num_filters*2, kernel_size=(1,1,1), activation='elu', padding='same')(fc2)
	d2=Dropout(0.2)(fc3)
	# Output layers
	f1=Flatten()(d2)
	#output1=Dense(21, activation='linear')(f1)
	output1=Dense(21, activation='linear')(f1)
	con1=concatenate([output1, input3])
	output2=Dense(42, activation='elu')(con1)
	output2=Dense(21, activation='linear',name='output1')(output2)

	model = Model(inputs =[input1, input2], outputs=[output2])
	model.compile(optimizer = Adam(lr=5e-4), loss = {'output1':'mean_squared_logarithmic_error'}, metrics = {'output1':'mean_absolute_error'}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef SGD(lr=0.01)

	return model

if __name__ == '__main__':
    model = Unet()
    model.summary()
    plot_model(model, to_file='model_plot_unet.png', show_shapes=True, show_layer_names=True)
