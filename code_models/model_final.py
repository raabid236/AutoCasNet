# Kibrom berihu Girum
# 10/08/2018
# U-net for sgementation of radioactive seeds in prostate post-implant brachytherapy 
# This script is to read, save and load data (train, mask and test) 
# MODEL UNET 
import os
import sys
sys.path.append('E:/Dev/code_organized/Seed_net/')
import numpy as np 
from keras.models import Model 
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation, Dropout , Add 
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, add, RNN 
from keras.optimizers import Adam 
from keras import backend as K
import train_agumnted_only as mtp 
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU

from keras.utils.vis_utils import plot_model
# Define the parameters of the data to process
K.set_image_data_format('channels_last')
img_cols = 256
img_rows = 256
img_channels = 1

def convolution_block(x, num_filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(num_filters, size, strides=strides, kernel_initializer='he_normal', padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    return x

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
	se = GlobalMaxPooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
	se = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(se)
	if K.image_data_format() == 'channels_first':
		se = Permute((3, 1, 2))(se)
	init = Conv2D(1, (1, 1), padding='same')(init) # change of size 
	x = multiply([init, se])
	return x


def senet_block(blockInput, num_filters, size, strides=(1,1), padding='same', activation=True):
    x = convolution_block(blockInput, num_filters, size=(3,3), strides=strides, padding=padding, activation=True)
    x=se_block(x,filters=num_filters, ratio=8)
    x = convolution_block(x, num_filters, size=(3,3), strides=strides, padding=padding, activation=True)
    
    return x


def unet_block(blockInput, num_filters, size, strides=(1,1), padding='same', activation=True):
    x = convolution_block(blockInput, num_filters, size=(3,3), strides=strides, padding=padding, activation=True)
    x = convolution_block(x, num_filters, size=(3,3), strides=strides, padding=padding, activation=True)
    
    return x


def residual_block(blockInput, num_filters, size, strides=(1,1), padding='same', activation=True):
    x = convolution_block(blockInput, num_filters, size=(3,3), strides=strides, padding=padding, activation=True)
    blockInput1= Conv2D(num_filters, (1, 1), padding='same')(blockInput) # change of size
    x = Add()([x, blockInput1])
    x = convolution_block(x, num_filters, size=(3,3), strides=strides, padding=padding, activation=True)

    #x2 = convolution_block(blockInput, num_filters, size=(5,5), strides=strides, padding=padding, activation=True)
    #x3 = convolution_block(blockInput, num_filters, size=(1,1), strides=strides, padding=padding, activation=True)
    #x = concatenate([x, x2, x3])
    #x = GlobalMaxPooling2D()(x)
    #x2 = convolution_block(blockInput, num_filters, size=(5,5), strides=strides, padding=padding, activation=True)
    #blockInput2= Conv2D(num_filters, (1, 1), padding='same')(blockInput) # change of size 
    #x = Add()([x2, blockInput2])

    #x3 = convolution_block(blockInput, num_filters, size=(5,5), strides=strides, padding=padding, activation=True)
    #x = concatenate([x, x2])
    #x = concatenate([x, x3])

    #blockInput1= Conv2D(num_filters, (1, 1), padding='same')(x) # change of size 
    #x = Add()([x, blockInput1])

    return x

def res_unet(num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')

    c1 = residual_block(inputs, num_filters, size=(3, 3))
    #c1=se_block(c1, filters=num_filters, ratio=16)
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    #p1 = Dropout(0.2)(p1)

    c2 = residual_block(p1, num_filters*2, size=(3, 3))
    #c2=se_block(c2, filters=num_filters*2, ratio=16)
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    #p2 = Dropout(0.2)(p2)
    
    c3 = residual_block(p2, num_filters*4, size=(3, 3))
    #c3=se_block(c3, filters=num_filters*4, ratio=16)
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    #p3 = Dropout(0.2)(p3)

    #c4 = residual_block(p3, num_filters*8, size=(3, 3))
    #c4=se_block(c4, filters=16*8, ratio=16)
    #p4 = MaxPooling2D(pool_size=(2,2))(c4)
    #p4 = Dropout(0.2)(p4)

    #c5 = residual_block(p4, num_filters*16, size=(3, 3))
    #c5=se_block(c5, filters=16*16, ratio=16)
    #p5 = MaxPooling2D(pool_size=(2,2))(c5)

    # # bottelneck
    c6 = residual_block(p3, num_filters*8, size=(3, 3))
    #c6=se_block(c6, filters=num_filters*8, ratio=16)
    #c6 = residual_block(p3, num_filters*8, size=(3, 3))
    #c6=se_block(c6, filters=16*32, ratio=16)
    c6 = Dropout(0.2)(c6)
  
    # upsamplying with concatination 
    #u1 = Conv2DTranspose(num_filters*16, (2,2), strides=(2,2), padding='same')(c6)
    #u1= concatenate([u1, c5])
    #c7 = convolution_block(u1, num_filters*16, size=(3,3))
    #c7 = convolution_block(c7, num_filters*16, size=(3,3))

    # upsamplying with concatination 
    #u2 = Conv2DTranspose(num_filters*8, (2,2), strides=(2,2), padding='same')(c6)
    #u2 = concatenate([u2, c4])
    #u2 = Dropout(0.2)(u2)
    #c8 = convolution_block(u2, num_filters*8, size=(3,3))
    #c8 = convolution_block(c8, num_filters*8, size=(3,3))

    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    #u3 = Dropout(0.2)(u3)
    #c9 = convolution_block(u3, num_filters*4, size=(3,3))
    #c9 = se_block(c9, filters=num_filters*4, ratio=16)
    #c9 = convolution_block(c9, num_filters*4, size=(3,3))
    c9 = residual_block(u3, num_filters*4, size=(3, 3))

    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    #u4 = Dropout(0.2)(u4)
    #c10 = convolution_block(u4, num_filters*2, size=(3,3))
    #c10=se_block(c10, filters=num_filters*2, ratio=16)
    #c10 = convolution_block(c10, num_filters*2, size=(3,3))
    c10 = residual_block(u4, num_filters*2, size=(3, 3))

    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    #u5 = Dropout(0.2)(u5)
    #c11 = convolution_block(u5, num_filters, size=(3,3))
    #c11=se_block(c11, filters=num_filters, ratio=16)
    #c11= convolution_block(c11, num_filters,size= (3,3))
    c11 = residual_block(u5, num_filters, size=(3, 3))

    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)

    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss


def unet(num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')
    
    c1 = unet_block(inputs, num_filters, size=(3, 3))
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    
    c2 = unet_block(p1, num_filters*2, size=(3, 3))
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    
    c3 = unet_block(p2, num_filters*4, size=(3, 3))
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    
    # # bottelneck
    c6 = unet_block(p3, num_filters*8, size=(3, 3))
    c6 = Dropout(0.2)(c6)
    
    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    c9 = unet_block(u3, num_filters*4, size=(3, 3))
    
    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    c10 = unet_block(u4, num_filters*2, size=(3, 3))
    
    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    c11 = unet_block(u5, num_filters, size=(3, 3))
    
    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)
    
    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss


def resunet(num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')
    
    c1 = residual_block(inputs, num_filters, size=(3, 3))
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    
    c2 = residual_block(p1, num_filters*2, size=(3, 3))
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    
    c3 = residual_block(p2, num_filters*4, size=(3, 3))
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    
    # # bottelneck
    c6 = unet_block(p3, num_filters*8, size=(3, 3))
    c6 = Dropout(0.2)(c6)
    
    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    c9 = residual_block(u3, num_filters*4, size=(3, 3))
    
    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    c10 = residual_block(u4, num_filters*2, size=(3, 3))
    
    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    c11 = residual_block(u5, num_filters, size=(3, 3))
    
    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)
    
    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss

def autocasnet(img_channels=1, num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')
    
    c1 = residual_block(inputs, num_filters, size=(3, 3))
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    
    c2 = residual_block(p1, num_filters*2, size=(3, 3))
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    
    c3 = residual_block(p2, num_filters*4, size=(3, 3))
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    
    # # bottelneck
    c6 = unet_block(p3, num_filters*8, size=(3, 3))
    c6 = Dropout(0.2)(c6)
    
    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    c9 = residual_block(u3, num_filters*4, size=(3, 3))
    
    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    c10 = residual_block(u4, num_filters*2, size=(3, 3))
    
    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    c11 = residual_block(u5, num_filters, size=(3, 3))
    
    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)
    
    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss
    return model

def autocasnet1(img_channels=1, num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')
    
    c1 = unet_block(inputs, num_filters, size=(3, 3))
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    
    c2 = unet_block(p1, num_filters*2, size=(3, 3))
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    
    c3 = unet_block(p2, num_filters*4, size=(3, 3))
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    
    # # bottelneck
    c6 = unet_block(p3, num_filters*8, size=(3, 3))
    c6 = Dropout(0.2)(c6)
    
    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    c9 = unet_block(u3, num_filters*4, size=(3, 3))
    
    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    c10 = unet_block(u4, num_filters*2, size=(3, 3))
    
    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    c11 = unet_block(u5, num_filters, size=(3, 3))
    
    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)
    
    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss
    return model

def autocasnet2(img_channels=1, num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')
    
    c1 = senet_block(inputs, num_filters, size=(3, 3))
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    
    c2 = senet_block(p1, num_filters*2, size=(3, 3))
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    
    c3 = senet_block(p2, num_filters*4, size=(3, 3))
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    
    # # bottelneck
    c6 = unet_block(p3, num_filters*8, size=(3, 3))
    c6 = Dropout(0.2)(c6)
    
    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    c9 = senet_block(u3, num_filters*4, size=(3, 3))
    
    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    c10 = senet_block(u4, num_filters*2, size=(3, 3))
    
    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    c11 = senet_block(u5, num_filters, size=(3, 3))
    
    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)
    
    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss
    return model


def senet(img_channels=1, num_filters=32):
    inputs = Input((img_rows, img_cols, img_channels), name='input_one')
    
    c1 = senet_block(inputs, num_filters, size=(3, 3))
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    
    c2 = senet_block(p1, num_filters*2, size=(3, 3))
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    
    c3 = senet_block(p2, num_filters*4, size=(3, 3))
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    
    # # bottelneck
    c6 = unet_block(p3, num_filters*8, size=(3, 3))
    c6 = Dropout(0.2)(c6)
    
    u3 = Conv2DTranspose(num_filters*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = concatenate([u3, c3])
    c9 = senet_block(u3, num_filters*4, size=(3, 3))
    
    u4 = Conv2DTranspose(num_filters*2, (2,2), strides=(2,2), padding='same')(c9)
    u4 = concatenate([u4, c2])
    c10 = senet_block(u4, num_filters*2, size=(3, 3))
    
    u5 = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='same')(c10)
    u5 = concatenate([u5, c1])
    c11 = senet_block(u5, num_filters, size=(3, 3))
    
    ouput_layer= Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='output_mask')(c11)
    
    model = Model(inputs =[inputs], outputs=[ouput_layer])
    model.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])  # mtp.dice_coefficient, mtp.cross_dice_loss
    return model

if __name__ == '__main__':
    model=res_unet()
    model.summary()
    plot_model(model, to_file='model_plot_res_unet.png', show_shapes=True, show_layer_names=True)
