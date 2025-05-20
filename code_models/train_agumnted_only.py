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
import res_fcnn as fcnn
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt 

from keras.callbacks import TensorBoard

from datetime import datetime

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

def data_agumentationXYZ(img_train, mask_train):
	img_train = np.asarray(img_train)
	mask_train = np.asarray(mask_train)
	print ("Before Size")
	print(img_train.shape)
	print(mask_train.shape)

	img=np.zeros((3*img_train.shape[0],img_train.shape[1],img_train.shape[2],img_train.shape[3]))
	mask=np.zeros((3*mask_train.shape[0],mask_train.shape[1],mask_train.shape[2],mask_train.shape[3]))
	
	#img=img.reshape(int(img.shape[0]/256),img.shape[1],img.shape[1],img.shape[2],img.shape[3])
	#mask=mask.reshape(int(mask.shape[0]/256),mask.shape[1],mask.shape[1],mask.shape[2],mask.shape[3])
	img_train=img_train.reshape(int(img_train.shape[0]/256),img_train.shape[1],img_train.shape[1],img_train.shape[2],img_train.shape[3])
	mask_train=mask_train.reshape(int(mask_train.shape[0]/256),mask_train.shape[1],mask_train.shape[1],mask_train.shape[2],mask_train.shape[3])
	print(img_train.shape)
	print(mask_train.shape)
	print(img.shape)
	print(mask.shape)

	for i in range(0,img_train.shape[0]):
		print(i)
		for j in range(0,img_train.shape[1]):
			img[i*256*3+j,:,:,:]=np.copy(img_train[i,j,:,:,:])
			mask[i*256*3+j,:,:,:]=np.copy(mask_train[i,j,:,:,:])
		for j in range(0,img_train.shape[1]):
			img[i*256*3+j,:,:,:]=np.copy(img_train[i,:,j,:,:])
			mask[i*256*3+j,:,:,:]=np.copy(mask_train[i,:,j,:,:])
		for j in range(0,img_train.shape[1]):
			img[i*256*3+j,:,:,:]=np.copy(img_train[i,:,:,j,:])
			mask[i*256*3+j,:,:,:]=np.copy(mask_train[i,:,:,j,:])

	print('After resize')
	print(img.shape)
	print(mask.shape)
	return img, mask


def train_regression():
	print('-'*20)
	print('Loading and preprocessing train data...')
	print('-'*20)
	imgs_train, imgs_mask_train = here_load_train_data() 
	imgs_train = imgs_train.astype('float32')


	#Normalization 
	mean_train = np.mean(imgs_train[:14*256,:,:,:]) # mean for the data centering 
	std_train = np.std(imgs_train[:14*256,:,:,:])  # std for data normalization 
	imgs_train -= mean_train
	imgs_train /= std_train
	imgs_mask_train[imgs_mask_train>0] = 1

	# Note: As we use sigmoid activation function as output, consider we have an ouput between [0, 1]
	images_test=imgs_train[14*256:,:,:,:]
	mask_test=imgs_mask_train[14*256:,:,:,:]
	
	imgs_train1=imgs_train[:14*256,:,:,:]
	imgs_mask_train1=imgs_mask_train[:14*256,:,:,:]


	print('-'*20)
	print ('fitting model ....')
	print('-'*20)
	#imgs_train1, imgs_mask_train1 = data_agumentationXYZ(imgs_train1, imgs_mask_train1)
	model = rnet.res_unet()
	#model.load_weights('weights_regre_advise_20190313153916.h5')
	print(model.summary())
	earlystopper = EarlyStopping(patience=15, verbose=1)
	# earlystopper =EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
	model_ckeckpoint = ModelCheckpoint('weights_unet.h5', monitor ='val_loss', save_best_only =True)
	tensorboard_call = TensorBoard(log_dir ="logdir" + "/train_unet_{}".format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
	model.fit({'input_one':imgs_train1}, {'output_mask':imgs_mask_train1}, batch_size = 20 , epochs = 100, verbose = 1, shuffle =True, validation_split = 0.3, callbacks 		   =[tensorboard_call, model_ckeckpoint]) 
	
	#mask = model.predict({'input_one':imgs_train}, verbose=0)
	#np.save('predicted_maskCOR.npy', mask)
	#print('output saved')

	# serialize model to JSON
	model_json = model.to_json()
	with open("model_unet256.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model_unet256.h5")
	print("Saved model to disk")

	mask1 = model.predict({'input_one':imgs_train}, verbose=0)
	np.save('predicted_maskCOR.npy', mask1)
	print('output saved')
 
	#images_test = np.load('CochleaNormalizednew1/imagesCochlea256Test_768.npy')
	#images_test = np.expand_dims(images_test, axis=-1)
	#mask_test = np.load('CochleaNormalizednew1/gTCochlea256Test_768.npy')
	#mask_test = np.expand_dims(mask_test, axis=-1)
	#print(images_test.shape)
	#print(mask_test.shape)
	#~evaluate the model
	scores = model.evaluate({'input_one':imgs_train}, {'output_mask':imgs_mask_train}, verbose=1)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
	scores = model.evaluate({'input_one':images_test}, {'output_mask':mask_test}, verbose=1)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
	#scores = model.evaluate({'input_one':images_test}, {'output_mask':mask_test}, verbose=0)
	mask_pred = model.predict({'input_one':images_test}, verbose=0)
	np.save('predicted_maskTestUNET.npy', mask_pred)
	print('output saved')
	
# Assuming we have read the data and saved as .npy file 
def here_load_train_data():
	images_train = np.load('CochleaNormalizednew1/imagesCochlea256_21COR.npy')
	images_train = images_train[1024:,:,:]
	#images_train = np.load('CochleaNormalizednew1/imagesCochlea256Train_13824.npy')
	images_train = np.expand_dims(images_train, axis=-1)
	mask = np.load('CochleaNormalizednew1/gTCochlea256_21COR.npy')
	mask = mask[1024:,:,:]
	#mask = np.load('CochleaNormalizednew1/gTCochlea256Train_13824.npy')
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
