import os
from model import Deeplabv3,BilinearUpsampling,relu6
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras.regularizers import l2
from keras import optimizers, metrics
import cv2
import numpy as np
import itertools
import glob
from keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
from skimage.transform import resize
import scipy.io as sio
import  matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.losses import mean_squared_error
import math
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; # Choose the right GPU
SMOOTH = 1e-12
bs = 2

def getImageArr( path , width , height ):
	img=sio.loadmat(path)
	rgb = cv2.resize(img['rgb'],( width , height )) / 127.5 - 1
	depth = cv2.resize(img['depth'],( width , height ))
	depth = depth / 4.
	normals = cv2.resize(img['normals'],( width , height ))
	normals = normals
	img = np.dstack((rgb,depth,normals))
	return img

def getSegmentationArr( path , nClasses ,  width , height ):
	seg_labels = np.zeros((  height , width  , nClasses ))
	img = cv2.imread(path, 1)
	img = cv2.resize(img, ( width , height ))
	img = img[:, : , 0]
	for c in range(0,nClasses):
		seg_labels[: , : , c ] = (img == c ).astype(int)
	seg_labels = seg_labels[:,:,1:]
	mask = np.sum(seg_labels,-1)
	return seg_labels

def DataLoader(dir_img,dir_seg,batch_size):
	assert dir_img[-1] == '/'

	images = glob.glob( dir_img + "*.mat"  )
	images.sort()
	segmentations = glob.glob(dir_seg + '*.png')
	segmentations.sort()
	zipped = itertools.cycle( zip(images,segmentations))
	while True:
		X = []
		Y= []
		for _ in range( batch_size) :
			im, seg = next(zipped)
			X.append( getImageArr(im,560 , 425))
			y = getSegmentationArr_14( seg , 5 , 560 , 425)
			Y.append(y)
		yield np.array(X) , np.array(Y)

def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=False):
	""" this code is provided by Qubvel's repository: https://github.com/qubvel/segmentation_models"""
	r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
	(originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
	similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
	and is defined as the size of the intersection divided by the size of the union of the sample sets:
	.. math:: J(A, B) = \frac{A \cap B}{A \cup B}
	Args:
	gt: ground truth 4D keras tensor (B, H, W, C)
	pr: prediction 4D keras tensor (B, H, W, C)
	class_weights: 1. or list of class weights, len(weights) = C
	smooth: value to avoid division by zero
	per_image: if ``True``, metric is calculated as mean over images in batch (B),
	else over whole batch
	Returns:
	IoU/Jaccard score in range [0, 1]
	.. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index
	"""
	if per_image:
		axes = [1, 2]
	else:
		axes = [0, 1, 2]

	intersection = K.sum(gt * pr, axis=axes)
	union = K.sum((gt + pr)*mask, axis=axes) - intersection
	iou = (intersection + smooth) / (union + smooth)

		# mean per image
	if per_image:
		iou = K.mean(iou, axis=0)

			# weighted mean per class
		iou = K.mean(iou * class_weights)

	return iou

jaccard_score = iou_score

def jaccard_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    """ this code is provided by Qubvel's repository: https://github.com/qubvel/segmentation_models"""    
    r"""Jaccard loss function for imbalanced datasets:
    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
    Returns:
        Jaccard loss in range [0, 1]
    """
    return 1 - jaccard_score_4(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)



train_images_path = './TrainImages/'
train_segs_path = './TrainLabels/'
val_images_path = './ValImages/'
val_segs_path = './ValLabels/'
G1= DataLoader(train_images_path,train_segs_path,bs)
G2 = DataLoader(val_images_path,val_segs_path,bs)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule)


modelcp=keras.callbacks.ModelCheckpoint('./cp/{epoch:02d}-{val_pixel_accuracy_14:.3f}.h5',
                                          monitor='categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
model = Deeplabv3()
print(model.summary())
sgd = optimizers.SGD(lr=0.0, decay=0.0, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss =jaccard_loss,metrics = [metrics.categorical_accuracy])
lr_sched = step_decay_schedule(initial_lr=0.01, decay_factor=0.9, step_size=2)
history = model.fit_generator(G1,795//bs,validation_data=G2,validation_steps=654//bs,epochs=100,callbacks=[lr_sched,modelcp])
