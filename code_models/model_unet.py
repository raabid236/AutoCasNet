# Kibrom berihu Girum
# 15/02/2019
# U-net for sgementation of cells
# MODEL UNET 
# Import libraries to use 
import os
import sys
import numpy as np 
from keras.models import Model 
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Flatten, Dropout
from keras.optimizers import Adam 
from keras import backend as K
from keras.utils.vis_utils import plot_model
import pydot
import graphviz
# Define the parameters of the data to process, change according to your data_size 
img_cols = 256
img_rows = 256
img_channels = 1

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
	conv_feature1 = Conv2D(num_filters, kernel_size, activation='elu', padding='same')(input_image)
	conv_feature2 = Conv2D(num_filters, kernel_size, activation='elu', padding='same')(conv_feature1)
	return conv_feature2



def  Unet(num_filters=32):
	#input images of size [imgs_rows, img_cols, img_channels]
	inputs = Input((img_rows, img_cols,  img_channels), name ='input')

	#First convolution and pooling
	c1 = convolve_fnc(inputs, num_filters, kernel_size=(3,3))
	pool1 = MaxPooling2D(pool_size=(2,2))(c1)

	#Second convolution and pooling
	c2 = convolve_fnc(pool1, num_filters*2, kernel_size=(3,3))
	pool2 = MaxPooling2D(pool_size=(2,2))(c2)

	#Third convolution and pooling
	c3 = convolve_fnc(pool2, num_filters*4, kernel_size=(3,3))
	pool3 = MaxPooling2D(pool_size=(2,2))(c3)

	#Fourth convolution and pooling
	c4 = convolve_fnc(pool3, num_filters*8, kernel_size=(3,3))
	pool4 = MaxPooling2D(pool_size=(2,2))(c4)

	c5 = convolve_fnc(pool4, num_filters*8, kernel_size=(3,3))
	pool5 = MaxPooling2D(pool_size=(2,2))(c5)
	
	#Firth convolution and pooling
	c6 = convolve_fnc(pool5, num_filters*16, kernel_size=(3,3))
	c6=Dropout(0.2)(c6)

	u7 = Conv2DTranspose(num_filters*8, kernel_size=(2,2), strides=(2,2), padding='same')(c6)
	u7 = concatenate([u7, c5])
	c7 = convolve_fnc(u7, num_filters*8, kernel_size=(3,3))

	# upsamplying with concatination
	u8 = Conv2DTranspose(num_filters*8, kernel_size=(2,2), strides=(2,2), padding='same')(c7)
	u8 = concatenate([u8, c4])
	c8 = convolve_fnc(u8, num_filters*8, kernel_size=(3,3))

	u9 = Conv2DTranspose(num_filters*4, kernel_size=(2,2), strides=(2,2), padding='same')(c8)
	u9 = concatenate([u9, c3])
	c9 = convolve_fnc(u9, num_filters*4, kernel_size=(3,3))

	u10 = Conv2DTranspose(num_filters*2, kernel_size=(2,2), strides=(2,2), padding='same')(c9)
	u10 = concatenate([u10, c2])
	c10 = convolve_fnc(u10, num_filters*2, kernel_size=(3,3))

	u11 = Conv2DTranspose(num_filters, kernel_size=(2,2), strides=(2,2), padding='same')(c10)
	u11 = concatenate([u11, c1])
	c11 = convolve_fnc(u11, num_filters, kernel_size=(3,3))

	# Mask_output
	output_mask = Conv2D(1, (1, 1), activation='sigmoid', name='output_mask')(c11)

	model = Model(inputs =[inputs], outputs=[output_mask])
	model.compile(optimizer = Adam(lr=1e-4), loss = {'output_mask':dice_coef_loss}, metrics = {'output_mask':'accuracy'}) # 'binary_crossentropy', 'accuracy' or mtp.dice_coef_loss, mtp.dice_coef
	return model

if __name__ == '__main__':
    model = Unet()
    model.summary()
    plot_model(model, to_file='model_plot_unet.png', show_shapes=True, show_layer_names=True)
