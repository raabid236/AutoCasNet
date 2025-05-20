# Kibrom berihu Girum
# 10/08/2018
# U-net for sgementation of radioactive seeds in prostate post-implant brachytherapy 
# This script is to read, save and load data (train, mask and test) 

import os
import sys
import pandas as pd 
import random 
from skimage.io import imread, imsave
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf 
#import data_read_write.data_reading_nii_and_saving_npy as data_pro 
import res_unet as rnet
import model_final
import res_fcnn as fcnn
from keras.preprocessing.image import ImageDataGenerator
from nipy.labs.mask import largest_cc
import matplotlib.pyplot as plt 

from keras.callbacks import TensorBoard

from datetime import datetime

import os
import sys
import pandas as pd
import random
from skimage.io import imread, imsave
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
#import data_read_write.data_reading_nii_and_saving_npy as data_pro
import model_final as models
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from datetime import datetime

import math
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import Tensorboard_image
import model_unet
from keras import backend as K
from keras.losses import binary_crossentropy
import scipy as sp
from math import log
from skimage.transform import resize
from datetime import datetime
from keras.models import model_from_json
import time
import keras
from keras.callbacks import TensorBoard
from model_midl import midl
K.set_image_data_format('channels_last')
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from scipy.ndimage import zoom

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
import scipy.misc




K.set_image_data_format('channels_last')

# random seed for repeatability 
seed = 42
random.seed = seed
np.random.seed = seed

# Define the parameters of the data to process
img_cols =  256
img_rows = 256
image_channels = 1
smooth = 1.
# take train mean and std for testing too 
mean_train = 1
std_train = 1 
# Define metrics
def cross_dice_loss(y_true, y_pred):
	bcrox = K.binary_crossentropy(y_true, y_pred)
	dcloss =1-dice_coefficient(y_true, y_pred)
	return bcrox + dcloss
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
	loss = 1-dice_coefficient(y_true,y_pred )
	return loss 
def cros_entropy(y_true, y_pred):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	if y_true ==1:
		retun -log(y_pred)
	else:
		retun -log(1-y_pred)

def dice_coef(y_true, y_pred):
	# y_true=y_true[:,:,:2]
	# y_pred =y_pred[:,:,:2]
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	y_pred_f = K.cast(K.greater(y_pred_f, 0.5), 'float32')
	intersection = K.sum(y_true_f*y_pred_f)
	return (2.*intersection + smooth ) /  (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_loss(y_true, y_pred):
	# y_true_f  = tf.sigmoid(y_true)
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	# y_pred_f = K.cast(K.greater(y_pred_f, 0.5), 'float32')
	intersection = K.sum(y_true_f*y_pred_f)
	return 1- ((2.*intersection + smooth ) /  (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch
# Define IOU metric
def mean_iou(y_true, y_pred):
	prec =[]
	for t in np.arange(0.5, 1.0, 0.005):
		y_pred_ = tf.to_int32(y_pred >t)
		score, up_opt =tf.metrics.mean_iou(y_true, y_pred_, 2)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return K.mean(K.stack(prec), axis=0)

def preprocess(imgs, img_rows, img_cols): # to make the same rows and cols
	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype = np.uint8)
	for i in range (imgs.shape[0]):
		imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

	imgs_p = imgs_p [..., np.newaxis]
	return imgs_p 


def loss_box(y_true, y_pred):
	# output of the network 
	# Adjust predictions
	pred_box_confidence = K.softmax(y_pred[..., 0]) # existance of an object 
	pred_box_center = (y_pred[..., 1:15])  # cneter and maximum and minimum 

	# Adjust ground truth
	true_box_condfidence = y_true[...,0]
	true_box_center = y_true[..., 1:15]

	## Loss functions
	loss_conf = K.binary_crossentropy(true_box_condfidence, pred_box_confidence)
	# loss_conf = logloss_sklearn(true_box_condfidence, pred_box_confidence)

	loss_coord = loss_centroid_box(true_box_center, pred_box_center, true_box_condfidence )


	loss = 0.5*(loss_conf + loss_coord)

	return loss

def loss_centroid_box(y_true, y_pred, true_box_condfidence):
	#Note: positive 1, contibute to the loss functions
	# true_box_condfidence = K.squeeze(true_box_condfidence, -1)
	indices = tf.where(K.equal(true_box_condfidence, 1))
	# pick bbox that contribute to the loss
	y_pred_xy = tf.gather_nd(y_pred, indices)
	y_true_xy = tf.gather_nd(y_true, indices)
	'''
	# smooth l1 loss 
	loss = smooth_l1_loss(y_true_xy, y_pred_xy)
	loss = K.switch(tf.size(loss)>0, K.mean(loss), tf.constant(0.0))
	'''
	# mean squared error loss
	loss_mser = tf.losses.mean_squared_error(y_true_xy, y_pred_xy)
	

	return loss_mser


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def logloss(y_true, y_pred, eps = 1e-15):
	p = np.clip(y_pred, eps, 1-eps)
	if y_true == 1:
		return -log(p)
	else:
		return -log(1-p)
# from sklearn.metrics 
from sklearn.metrics import log_loss 

def logloss_sklearn(y_true, y_pred):
	loglos = log_loss(y_true, y_pred, eps=1e-15)
	return loglos

def iou(y_true, y_pred):
	"""
	Implement the intersection over union (IOU) between box1 and box2

	Arguments:
			box1: first box, list object with coordinates (x1, y1, x2, y2)
			box2: second box, list object with coordinates (x1, y1, x2, y2)

	"""
	box1 = y_true[..., 3:7]   # center, and min and max
	box2 = y_pred[..., 3:7] 
	print(box2.shape)
	# gettign the indices of true boxes
	true_box_condfidence = y_true[...,0]
	indices = tf.where(K.equal(true_box_condfidence, 1))
	# pick bbox that contribute to the loss of the width and height 
	box1 = tf.gather_nd(box1, indices)
	box2 = tf.gather_nd(box2, indices)

	# Calculate the (y1, x1, y2, x2) coordintes of the intersection of box1 and box2. calcuate its area. 
	xi1 = K.maximum(box1[0], box2[0])
	yi1 = K.maximum(box1[1], box2[1])
	xi2 = K.minimum(box1[2], box2[2])
	yi2 = K.minimum(box1[3], box2[3])
	inter_area = (xi1-xi2)*(yi1 - yi2)

	# Union formula: Union(A, B) = A+B - inter(A, B) 
	box1_area  = (box1[3] - box1[1]) *(box1[2] - box1[0])
	box2_area = (box2[3] - box2[1]) *(box2[2] - box2[0])
	union_area = box1_area + box2_area - inter_area


	# Compute the IOU
	iou = inter_area / union_area

	return iou 
def normalize(f):
	lmin = float(f.min())
	lmax = float(f.max())
	return np.floor((f-lmin)/(lmax-lmin)*0.5) 

def data_agumentation(img_train, mask_train):
	img_train = np.asarray(img_train)
	mask_train = np.asarray(mask_train)
	size_aug = 6
	print ("Before Size")
	print(img_train.shape)
	print(mask_train.shape)

	data_gen_args = dict(featurewise_center=False,
						featurewise_std_normalization=False,
						rescale = 128./255,
						rotation_range=45,
						shear_range=0.01, 
						zoom_range=0.5,
						horizontal_flip=True,
						vertical_flip=True,
						fill_mode='nearest')

	image_datagen = ImageDataGenerator(**data_gen_args)
	# mask_datagen = ImageDataGenerator(**data_gen_args)

	# Provide the same seed and keyword arguments to the fit and flow methods
	seed = 7
	image_datagen.fit(img_train, augment=True)
	# mask_datagen.fit(mask_train, augment=True)
	img =[]
	img =img_train
	mask = mask_train
	k=0;
	for X_batch, Y_mask in image_datagen.flow(img_train, mask_train, batch_size=size_aug):
		for i in range(0, size_aug):
			img = np.append(img, [X_batch[i]], axis=0)
			mask = np.append( mask, [Y_mask[i]], axis=0)
		k = k+1 
		if (k%10==0):
			print(k)
		if (k>=250):
			break

	print('After resize')
	print(img.shape)
	print(mask.shape)
	return img, mask

def get_callbacks(name_weights, patience_lr):
	mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=patience_lr, verbose=1, epsilon=1e-5, mode='min')
	earlystopper = EarlyStopping(patience=50, verbose=1)
	return [mcp_save, reduce_lr_loss, earlystopper]

def reset_weights(model):
	session = K.get_session()
	for layer in model.layers:
		if hasattr(layer, 'kernel_initializer'):
			layer.kernel.initializer.run(session=session)

def train_regression():
	print('-'*20)
	print('Loading and preprocessing train data...')
	print('-'*20)
	imgs_train, imgs_mask_train = here_load_train_data() 
	imgs_train = imgs_train.astype('float32')

    #Normalization
	mean_train = np.mean(imgs_train) # mean for the data centering
	std_train = np.std(imgs_train)  # std for data normalization
	imgs_train -= mean_train
	imgs_train /= std_train
	imgs_mask_train[imgs_mask_train>0] = 1

	k=4
	predOut1=np.zeros((imgs_mask_train.shape[0]*imgs_mask_train.shape[1],imgs_mask_train.shape[2],imgs_mask_train.shape[3],imgs_mask_train.shape[4]))
	predOut2=np.zeros((imgs_mask_train.shape[0]*imgs_mask_train.shape[1],imgs_mask_train.shape[2],imgs_mask_train.shape[3],imgs_mask_train.shape[4]))
	predOut3=np.zeros((imgs_mask_train.shape[0]*imgs_mask_train.shape[1],imgs_mask_train.shape[2],imgs_mask_train.shape[3],imgs_mask_train.shape[4]))
	predEval1=np.zeros((k,2))
	predEval2=np.zeros((k,2))
	predEval3=np.zeros((k,2))
	print(predOut3.shape)
    
	# Note: As we use sigmoid activation function as output, consider we have an ouput between [0, 1]
	for f in range(0,k):
		print('\nFold ',f)
		imgs_train1=np.copy(imgs_train)
		imgs_mask_train1=np.copy(imgs_mask_train)
		if f ==25:
			continue
		print('-'*20)
		print ('fitting models....')
		print('-'*20)
		####################################
		# Cascaded 1
		####################################
		print('-'*20)
		print ('fitting model 1....')
		print('-'*20)

		imgT=np.zeros((imgs_train1.shape[0]*imgs_train1.shape[1],imgs_train1.shape[2],imgs_train1.shape[3],1))
		imgT = imgT.astype('float32')
		mskT=np.zeros((imgs_train1.shape[0]*imgs_train1.shape[1],imgs_train1.shape[2],imgs_train1.shape[3],imgs_train1.shape[4]))
        
		for i in range(0,imgs_train1.shape[0]):
			for j in range(0,imgs_train1.shape[1]):
				imgT[i*imgs_train1.shape[1]+j,:,:,0]=np.copy(imgs_train1[i,:,:,j,0])
				mskT[i*imgs_mask_train1.shape[1]+j,:,:,:]=np.copy(imgs_mask_train1[i,:,:,j,:])
		print(imgT.shape)
		print(mskT.shape)
        
		if f==0:
			X_train_cv = np.copy(imgT[4*256:,:,:,:])
			y_train_cv = np.copy(mskT[4*256:,:,:,:])
			X_valid_cv = np.copy(imgT[:4*256,:,:,:])
			y_valid_cv= np.copy(mskT[:4*256,:,:,:])
		if f==1:
			X_train_cv1 = np.copy(imgT[:4*256,:,:,:])
			y_train_cv1 = np.copy(mskT[:4*256,:,:,:])
			X_train_cv2 = np.copy(imgT[8*256:,:,:,:])
			y_train_cv2 = np.copy(mskT[8*256:,:,:,:])
			X_train_cv = np.concatenate([X_train_cv1,X_train_cv2],axis=0)
			y_train_cv = np.concatenate([y_train_cv1,y_train_cv2],axis=0)

			#X_train_cv = np.copy(imgT[np.r_[:4*256,8*256:],:,:,:])
			#y_train_cv = np.copy(mskT[np.r_[:4*256,8*256:],:,:,:])
			X_valid_cv = np.copy(imgT[4*256:8*256,:,:,:])
			y_valid_cv= np.copy(mskT[4*256:8*256,:,:,:])
		if f==2:
			X_train_cv1 = np.copy(imgT[:8*256,:,:,:])
			y_train_cv1 = np.copy(mskT[:8*256,:,:,:])
			X_train_cv2 = np.copy(imgT[12*256:,:,:,:])
			y_train_cv2 = np.copy(mskT[12*256:,:,:,:])
			X_train_cv = np.concatenate([X_train_cv1,X_train_cv2],axis=0)
			y_train_cv = np.concatenate([y_train_cv1,y_train_cv2],axis=0)

			#X_train_cv = np.copy(imgT[np.r_[:8*256,12*256:],:,:,:])
			#y_train_cv = np.copy(mskT[np.r_[:8*256,12*256:],:,:,:])
			X_valid_cv = np.copy(imgT[8*256:12*256,:,:,:])
			y_valid_cv= np.copy(mskT[8*256:12*256,:,:,:])
		if f==3:
			X_train_cv = np.copy(imgT[:12*256,:,:,:])
			y_train_cv = np.copy(mskT[:12*256,:,:,:])
			X_valid_cv = np.copy(imgT[12*256:,:,:,:])
			y_valid_cv= np.copy(mskT[12*256:,:,:,:])
    
		print(X_train_cv.shape)
		print(y_train_cv.shape)
		print(X_valid_cv.shape)
		print(y_valid_cv.shape)

		if f==0:
			scipy.misc.imsave('model1fig1a.jpg', X_train_cv[120,:,:,0])
			scipy.misc.imsave('model1fig1b.jpg', y_train_cv[120,:,:,0])
			print('model 1 images saved')

		model1 = model_final.autocasnet1(img_channels=1)
		#print(model1.summary())
		name_weights = "shape_model1_autocasnet_4fold_" + str(f) + "_weights.h5"
		callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
		model1.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])
		model1.fit({'input_one':X_train_cv}, {'output_mask':y_train_cv}, batch_size =20, epochs = 75, verbose =1, shuffle =True, validation_data =(X_valid_cv, y_valid_cv), callbacks = callbacks)
		predEval1[f,:]=model1.evaluate(X_valid_cv, y_valid_cv)
		print(predEval1)

		if f==0:
			predOut1[:4*256,:,:,:]=model1.predict(X_valid_cv)
		if f==1:
			predOut1[4*256:8*256,:,:,:]=model1.predict(X_valid_cv)
		if f==2:
			predOut1[8*256:12*256,:,:,:]=model1.predict(X_valid_cv)
		if f==3:
			predOut1[12*256:,:,:,:]=model1.predict(X_valid_cv)

		# serialize model to JSON
		model_json = model1.to_json()
		with open("model_autocas1.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model1.save_weights("model_autocas1.h5")
		print("Saved model 1 to disk")

		mask2 = model1.predict({'input_one':imgT}, verbose=0)




		####################################
		# Cascaded 2
		####################################
		print('-'*20)
		print ('fitting model 2....')
		print('-'*20)
		######################################################################################
		maskTemp=np.copy(imgs_train1)
	
		for i in range(0,imgs_train1.shape[0]):
			for j in range(0,imgs_train1.shape[1]):
				maskTemp[i,:,:,j,0]=(mask2[i*imgs_train1.shape[1]+j,:,:,0])
		maskTemp[maskTemp>=0.5]=1
		maskTemp[maskTemp<0.5]=0
		for i in range(0,maskTemp.shape[0]):
			maskTemp[i,:,:,:,:]=largest_cc(maskTemp[i,:,:,:,:])
		mask2=np.copy(maskTemp)
		######################################################################################
		mask3=np.copy(mask2)
		######################################################################################
		imgT=np.zeros((imgs_train1.shape[0]*imgs_train1.shape[1],imgs_train1.shape[2],imgs_train1.shape[3],2))
		imgT = imgT.astype('float32')
		mskT=np.zeros((imgs_train1.shape[0]*imgs_train1.shape[1],imgs_train1.shape[2],imgs_train1.shape[3],imgs_train1.shape[4]))

		for i in range(0,imgs_train1.shape[0]):
			for j in range(0,imgs_train1.shape[1]):
				imgT[i*imgs_train1.shape[1]+j,:,:,0]=np.copy(imgs_train1[i,:,j,:,0])
				imgT[i*imgs_train1.shape[1]+j,:,:,1]=np.copy(mask2[i,:,j,:,0])
				mskT[i*imgs_mask_train1.shape[1]+j,:,:,:]=np.copy(imgs_mask_train1[i,:,j,:,:])
		print(imgT.shape)
		print(mskT.shape)
		imgT = imgT.astype('float32')

		if f==0:
			X_train_cv = np.copy(imgT[4*256:,:,:,:])
			y_train_cv = np.copy(mskT[4*256:,:,:,:])
			X_valid_cv = np.copy(imgT[:4*256,:,:,:])
			y_valid_cv= np.copy(mskT[:4*256,:,:,:])
		if f==1:
			X_train_cv1 = np.copy(imgT[:4*256,:,:,:])
			y_train_cv1 = np.copy(mskT[:4*256,:,:,:])
			X_train_cv2 = np.copy(imgT[8*256:,:,:,:])
			y_train_cv2 = np.copy(mskT[8*256:,:,:,:])
			X_train_cv = np.concatenate([X_train_cv1,X_train_cv2],axis=0)
			y_train_cv = np.concatenate([y_train_cv1,y_train_cv2],axis=0)
			#X_train_cv = np.copy(imgT[np.r_[:4*256,8*256:],:,:,:])
			#y_train_cv = np.copy(mskT[np.r_[:4*256,8*256:],:,:,:])
			X_valid_cv = np.copy(imgT[4*256:8*256,:,:,:])
			y_valid_cv= np.copy(mskT[4*256:8*256,:,:,:])
		if f==2:
			X_train_cv1 = np.copy(imgT[:8*256,:,:,:])
			y_train_cv1 = np.copy(mskT[:8*256,:,:,:])
			X_train_cv2 = np.copy(imgT[12*256:,:,:,:])
			y_train_cv2 = np.copy(mskT[12*256:,:,:,:])
			X_train_cv = np.concatenate([X_train_cv1,X_train_cv2],axis=0)
			y_train_cv = np.concatenate([y_train_cv1,y_train_cv2],axis=0)
			#X_train_cv = np.copy(imgT[np.r_[:8*256,12*256:],:,:,:])
			#y_train_cv = np.copy(mskT[np.r_[:8*256,12*256:],:,:,:])
			X_valid_cv = np.copy(imgT[8*256:12*256,:,:,:])
			y_valid_cv= np.copy(mskT[8*256:12*256,:,:,:])
		if f==3:
			X_train_cv = np.copy(imgT[:12*256,:,:,:])
			y_train_cv = np.copy(mskT[:12*256,:,:,:])
			X_valid_cv = np.copy(imgT[12*256:,:,:,:])
			y_valid_cv= np.copy(mskT[12*256:,:,:,:])


		if f==0:
			scipy.misc.imsave('model2fig1a.jpg', X_train_cv[120,:,:,0])
			scipy.misc.imsave('model2fig1b.jpg', y_train_cv[120,:,:,0])
			scipy.misc.imsave('model2fig1c.jpg', X_train_cv[120,:,:,1])
			print('model 2 images saved')


		model2 = model_final.autocasnet1(img_channels=2)
		#print(model2.summary())
		name_weights = "shape_model2_autocasnet_4fold_" + str(f) + "_weights.h5"
		callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
		model2.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])
		model2.fit({'input_one':X_train_cv}, {'output_mask':y_train_cv}, batch_size =20, epochs = 75, verbose =1, shuffle =True, validation_data =(X_valid_cv, y_valid_cv), callbacks = callbacks)
		predEval2[f,:]=model2.evaluate(X_valid_cv, y_valid_cv)
		print(predEval2)
        
		if f==0:
			predOut2[:4*256,:,:,:]=model2.predict(X_valid_cv)
		if f==1:
			predOut2[4*256:8*256,:,:,:]=model2.predict(X_valid_cv)
		if f==2:
			predOut2[8*256:12*256,:,:,:]=model2.predict(X_valid_cv)
		if f==3:
			predOut2[12*256:,:,:,:]=model2.predict(X_valid_cv)
        
		# serialize model to JSON
		model_json = model2.to_json()
		with open("model_autocas2.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model2.save_weights("model_autocas2.h5")
		print("Saved model 2 to disk")
        
		mask2 = model2.predict({'input_one':imgT}, verbose=0)

		####################################
		# Cascaded 3
		####################################
		print('-'*20)
		print ('fitting model 3...')
		print('-'*20)
		######################################################################################
		maskTemp=np.copy(imgs_train1)
	
		for i in range(0,imgs_train1.shape[0]):
			for j in range(0,imgs_train1.shape[1]):
				maskTemp[i,:,j,:,0]=(mask2[i*imgs_train1.shape[1]+j,:,:,0])
		maskTemp[maskTemp>=0.5]=1
		maskTemp[maskTemp<0.5]=0
		for i in range(0,maskTemp.shape[0]):
			maskTemp[i,:,:,:,:]=largest_cc(maskTemp[i,:,:,:,:])
		mask2=np.copy(maskTemp)
		######################################################################################
		imgT=np.zeros((imgs_train1.shape[0]*imgs_train1.shape[1],imgs_train1.shape[2],imgs_train1.shape[3],3))
		imgT = imgT.astype('float32')
		mskT=np.zeros((imgs_train1.shape[0]*imgs_train1.shape[1],imgs_train1.shape[2],imgs_train1.shape[3],imgs_train1.shape[4]))

		for i in range(0,imgs_train1.shape[0]):
			for j in range(0,imgs_train1.shape[1]):
				imgT[i*imgs_train1.shape[1]+j,:,:,0]=np.copy(imgs_train1[i,j,:,:,0])
				imgT[i*imgs_train1.shape[1]+j,:,:,1]=np.copy(mask2[i,j,:,:,0])
				imgT[i*imgs_train1.shape[1]+j,:,:,2]=np.copy(mask3[i,j,:,:,0])
				mskT[i*imgs_mask_train1.shape[1]+j,:,:,:]=np.copy(imgs_mask_train1[i,j,:,:,:])
		print(imgT.shape)
		print(mskT.shape)
		imgT = imgT.astype('float32')

		if f==0:
			X_train_cv = np.copy(imgT[4*256:,:,:,:])
			y_train_cv = np.copy(mskT[4*256:,:,:,:])
			X_valid_cv = np.copy(imgT[:4*256,:,:,:])
			y_valid_cv= np.copy(mskT[:4*256,:,:,:])
		if f==1:
			X_train_cv1 = np.copy(imgT[:4*256,:,:,:])
			y_train_cv1 = np.copy(mskT[:4*256,:,:,:])
			X_train_cv2 = np.copy(imgT[8*256:,:,:,:])
			y_train_cv2 = np.copy(mskT[8*256:,:,:,:])
			X_train_cv = np.concatenate([X_train_cv1,X_train_cv2],axis=0)
			y_train_cv = np.concatenate([y_train_cv1,y_train_cv2],axis=0)
			#X_train_cv = np.copy(imgT[np.r_[:4*256,8*256:],:,:,:])
			#y_train_cv = np.copy(mskT[np.r_[:4*256,8*256:],:,:,:])
			X_valid_cv = np.copy(imgT[4*256:8*256,:,:,:])
			y_valid_cv= np.copy(mskT[4*256:8*256,:,:,:])
		if f==2:
			X_train_cv1 = np.copy(imgT[:8*256,:,:,:])
			y_train_cv1 = np.copy(mskT[:8*256,:,:,:])
			X_train_cv2 = np.copy(imgT[12*256:,:,:,:])
			y_train_cv2 = np.copy(mskT[12*256:,:,:,:])
			X_train_cv = np.concatenate([X_train_cv1,X_train_cv2],axis=0)
			y_train_cv = np.concatenate([y_train_cv1,y_train_cv2],axis=0)
			#X_train_cv = np.copy(imgT[np.r_[:8*256,12*256:],:,:,:])
			#y_train_cv = np.copy(mskT[np.r_[:8*256,12*256:],:,:,:])
			X_valid_cv = np.copy(imgT[8*256:12*256,:,:,:])
			y_valid_cv= np.copy(mskT[8*256:12*256,:,:,:])
		if f==3:
			X_train_cv = np.copy(imgT[:12*256,:,:,:])
			y_train_cv = np.copy(mskT[:12*256,:,:,:])
			X_valid_cv = np.copy(imgT[12*256:,:,:,:])
			y_valid_cv= np.copy(mskT[12*256:,:,:,:])

		if f==0:
			scipy.misc.imsave('model3fig1a.jpg', X_train_cv[120,:,:,0])
			scipy.misc.imsave('model3fig1b.jpg', y_train_cv[120,:,:,0])
			scipy.misc.imsave('model3fig1c.jpg', X_train_cv[120,:,:,1])
			scipy.misc.imsave('model3fig1d.jpg', X_train_cv[120,:,:,2])
			print('model 3 images saved')


		model3 = model_final.autocasnet1(img_channels=3)
		#print(model3.summary())
		name_weights = "shape_model3_autocasnet_4fold_" + str(f) + "_weights.h5"
		callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
		model3.compile(optimizer = Adam(lr=1e-4), loss = mtp.cross_dice_loss, metrics = [mtp.dice_coef])
		model3.fit({'input_one':X_train_cv}, {'output_mask':y_train_cv}, batch_size =20, epochs = 75, verbose =1, shuffle =True, validation_data =(X_valid_cv, y_valid_cv), callbacks = callbacks)
		predEval3[f,:]=model3.evaluate(X_valid_cv, y_valid_cv)
		print(predEval3)

		if f==0:
			predOut3[:4*256,:,:,:]=model3.predict(X_valid_cv)
		if f==1:
			predOut3[4*256:8*256,:,:,:]=model3.predict(X_valid_cv)
		if f==2:
			predOut3[8*256:12*256,:,:,:]=model3.predict(X_valid_cv)
		if f==3:
			predOut3[12*256:,:,:,:]=model3.predict(X_valid_cv)
        
		# serialize model to JSON
		model_json = model3.to_json()
		with open("model_autocas3.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model3.save_weights("model_autocas3.h5")
		print("Saved model 3 to disk")

		mask2 = model3.predict({'input_one':imgT}, verbose=0)

		###############################

	np.save('predicted_autocasnet1_4fold.npy', predOut1)
	np.save('predicted_autocasnet2_4fold.npy', predOut2)
	np.save('predicted_autocasnet3_4fold.npy', predOut3)
	print("Outputs saved")


	
# Assuming we have read the data and saved as .npy file 
def here_load_train_data():
	images_train = np.load('CochleaNormalizednew1/imagesCochlea17_4352.npy')
	images_train=images_train.reshape(int(images_train.shape[0]/images_train.shape[1]),images_train.shape[1],images_train.shape[1],images_train.shape[2])
	images_train = np.expand_dims(images_train, axis=-1)
	mask = np.load('CochleaNormalizednew1/gTCochlea17_4352.npy')
	mask=mask.reshape(int(mask.shape[0]/mask.shape[1]),mask.shape[1],mask.shape[1],mask.shape[2])
	mask = np.expand_dims(mask, axis=-1)
	#mask[0:1024,:,:, :]=mask[mask.shape[0]-1024: ,:,:, :]
	#images_train[0:1024 , :, :,:] = images_train[images_train.shape[0]-1024: , :, :,:]
	#mask=mask[:mask.shape[0]-1024 ,:,:, :]
	#images_train = images_train[:images_train.shape[0]-1024 , :, :,:]
	print(images_train.shape)
	print(mask.shape)
	return images_train, mask
if __name__ == '__main__':

	# Train and predict u-net model
	train_regression()
