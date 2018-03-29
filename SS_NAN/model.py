"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
from __future__ import division
import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import scipy.ndimage
import tensorflow as tf
from tensorflow.python import debug as tfdbg
cfig=tf.ConfigProto(allow_soft_placement=True)
cfig.gpu_options.allow_growth=True
sess =tf.Session(config=cfig)

from .psp_resnet_builder import ResNet,ResNet_mutiout
from keras.backend.tensorflow_backend import set_session
set_session(sess)
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import scipy.ndimage
from .utils import (batch_slice, apply_box_deltas,
                    non_max_suppression, resize_corp_image_mask,
                    resize_mask,
                    unmold_mask_superpixel_filter)
from .utils import resize_image as utils_resize_image

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=None)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98+i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    return clipped

class SuperPixelFilterLayer(KE.Layer):
    def __init__(self,max_superpixel_num,config=None,**kwargs):
        super(SuperPixelFilterLayer, self).__init__(**kwargs)
        self.max_superpixel_num = max_superpixel_num
        self.config = config
    def call(self, inputs):
        feature_map = inputs[0]
        superpixel_map = inputs[1]
        f_shape = tf.shape(feature_map)
        channels = f_shape[3]
        s_shape = tf.shape(superpixel_map)
        feature_map = tf.image.resize_bilinear(feature_map,[s_shape[1],s_shape[2]])
        def superpixel_filter(f,s):
            x=tf.reshape(f,[-1,channels])
            y=tf.reshape(s,[-1])
            x=tf.unsorted_segment_sum(x,y,self.max_superpixel_num)
            weights=tf.cast(tf.bincount(y,minlength=self.max_superpixel_num,
                                        maxlength=self.max_superpixel_num,weights=None),tf.float32)+1e-6
            r=tf.gather(x/tf.expand_dims(weights,1),y)
            r=tf.reshape(r,[s_shape[1],s_shape[2],channels])
            return r
        r_feature_map = batch_slice([feature_map,superpixel_map],superpixel_filter,self.config.IMAGES_PER_GPU)
        #result = tf.image.resize_bilinear(r_feature_map,[f_shape[1],f_shape[2]])
        return r_feature_map
    def compute_output_shape(self, input_shape):
        return input_shape[1]+(input_shape[0][3],)
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)




def clip_to_window(window, boxes):
    """
    window: (y1, x1, y2, x2). The window in the image we want to clip to.
    boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], window[2]), window[0])
    boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], window[3]), window[1])
    boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], window[2]), window[0])
    boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], window[3]), window[1])
    return boxes


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    # Class IDs per ROI
    class_ids = np.argmax(probs, axis=1)
    # Class probability of the top class of each ROI
    class_scores = probs[np.arange(class_ids.shape[0]), class_ids]
    # Class-specific bounding box deltas
    deltas_specific = deltas[np.arange(deltas.shape[0]), class_ids]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Convert coordiates to image domain
    # TODO: better to keep them normalized until later
    height, width = config.IMAGE_SHAPE[:2]
    refined_rois *= np.array([height, width, height, width])
    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)
    # Round and cast to int since we're deadling with pixels now
    refined_rois = np.rint(refined_rois).astype(np.int32)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = np.where(class_ids > 0)[0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep = np.intersect1d(
            keep, np.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[0])

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = non_max_suppression(
            pre_nms_rois[ixs], pre_nms_scores[ixs],
            config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
    keep = np.intersect1d(keep, nms_keep).astype(np.int32)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = np.argsort(class_scores[keep])[::-1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = np.hstack((refined_rois[keep],
                        class_ids[keep][..., np.newaxis],
                        class_scores[keep][..., np.newaxis]))
    return result






def build_fpn_mask_graph(rois, feature_maps,
                         image_shape, pool_size, num_classes):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from diffent layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_shape: [height, width, depth]
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results

    Returns: Masks [batch, roi_count, height, width, num_classes]
    """
    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size], image_shape,
                        name="roi_align_mask")([rois] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3),
                           name='mrcnn_mask_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3),
                           name='mrcnn_mask_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3),
                           name='mrcnn_mask_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(axis=3),
                           name='mrcnn_mask_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2,2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x

############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1-less_than_one) * (diff - 0.5)
    return loss
def mrcnn_mask_loss_graph(target_masks, pred_masks,OHEM=True,KK=3):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.


    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss =K.categorical_crossentropy(target_masks,pred_masks)*K.sum(target_masks,-1)
    if OHEM:
        shape = tf.shape(loss)
        loss,_ = tf.nn.top_k(tf.reshape(loss,[-1,shape[1]*shape[2]]),k=tf.cast(shape[1]*shape[2]/KK,tf.int32),sorted=False)
    else:
        loss = K.sum(loss,[1,2])/K.sum(target_masks,[1,2,3])
    loss = K.mean(loss)
    loss = K.reshape(loss, [1, 1])
    return loss


############################################################
#  Data Generator
############################################################
def resize_image(image,config, min_dim=None, max_dim=None, padding=False,augment =False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1,1.0* min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = 1.0*max_dim / image_max
    # Resize image and mask
    if scale != 1:
        if augment:
            scale=scale*( np.random.random()*0.25+0.75  )
        image = scipy.misc.imresize(
            image, (int(round(h * scale)), int(round(w * scale))))
    # Need padding?

    image = mold_image(image.astype(np.float32),config)
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding
def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    bbox: [instance_count, (y1, x1, y2, x2, class_id)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask = dataset.load_mask(image_id)
    shape = image.shape
    if augment:
        image,mask = resize_corp_image_mask(image, np.expand_dims(mask,-1), config,min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        mask = np.squeeze(mask,-1)
        window =[0,0,config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM]
    else:
        image, window, scale, padding = utils_resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING,augment=False)
        mask = resize_mask(np.expand_dims(mask,-1), scale, padding)
        mask = np.squeeze(mask,-1)
    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            for v in range(3):
                a=(mask==14+2*v)
                b=(mask==14+2*v+1)
                mask[a]=14+2*v+1
                mask[b]=14+2*v

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[class_ids] = 1


    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    return image, image_meta, mask




def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2, class_id)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_meta,gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about

            # RPN Targets


            # Mask R-CNN Targets
            if b == 0:
                batch_image_meta = np.zeros((batch_size,)+image_meta.shape, dtype=image_meta.dtype)
                batch_images = np.zeros((batch_size,)+image.shape, dtype=np.float32)
                batch_gt_masks = np.zeros((batch_size, image.shape[0], image.shape[1]),dtype=np.uint8)

            # If more instances than fits in the array, sub-sample from them.
            # Add to batch
            batch_image_meta[b] = image_meta
            batch_images[b] = mold_image(image.astype(np.float32),config)
            batch_gt_masks[b,:,:] = gt_masks
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_gt_masks]
                outputs = []
                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise



############################################################
#  MaskRCNN Class
############################################################
class GRenderLayer(KE.Layer):
    def __init__(self,sigma,map_size,**kwargs):
        super(GRenderLayer,self).__init__(**kwargs)
        self.sigma = sigma
        self.map_size=map_size
    def call(self, inputs):
        centers = tf.expand_dims(inputs,-1)
        h=self.map_size[0]
        w=self.map_size[1]
        Wx,Wy =tf.meshgrid(tf.range(0,w),tf.range(0,h))
        Wx=tf.expand_dims(tf.expand_dims(Wx,0),0)
        Wy=tf.expand_dims(tf.expand_dims(Wy,0),0)
        Wx=tf.cast(Wx,tf.float32)
        Wy=tf.cast(Wy,tf.float32)
        Cy=tf.expand_dims(centers[...,0,:],-1)
        Cx=tf.expand_dims(centers[...,1,:],-1)
        Dx=tf.cast(Cx,tf.float32)-Wx
        Dy=tf.cast(Cy,tf.float32)-Wy
        result=4/(np.sqrt(np.pi*2)*self.sigma)*tf.exp(-0.5*(Dx*Dx+Dy*Dy)/(self.sigma**2))*tf.cast(tf.sign(tf.cast(Cx+Cy,tf.int32)),tf.float32)
        result=tf.transpose(result,[0,2,3,1])
        return result
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.map_size[0],self.map_size[1],input_shape[1])
class ComputeCentersLayer(KE.Layer):
    def __init__(self,num_classes,merge_dict=[0,1,1,0,1,2,0,2,0,3,0,2,3,1,4,5,6,7,8,9],**kwargs):
        super(ComputeCentersLayer,self).__init__(**kwargs)
        self.num_classes = num_classes
        self.merge_dict=np.array(merge_dict)
        assert self.num_classes==len(self.merge_dict),'num_classes must be equal to mergedict length'
    def call(self, inputs):
        heatmap = inputs
        h=tf.shape(heatmap)[1]
        w=tf.shape(heatmap)[2]
        Wx,Wy =tf.meshgrid(tf.range(0,w),tf.range(0,h))
        Wx=tf.expand_dims(tf.expand_dims(Wx,0),-1)
        Wy=tf.expand_dims(tf.expand_dims(Wy,0),-1)
        Wx=tf.cast(Wx,tf.float32)
        Wy=tf.cast(Wy,tf.float32)
        Dx=tf.split(heatmap*Wx,self.num_classes,-1)
        Dy=tf.split(heatmap*Wy,self.num_classes,-1)
        resultx=[]
        resulty=[]
        for v in range(1,np.max(self.merge_dict)+1):
            sx=[Dx[j] for j in np.nonzero(self.merge_dict==v)[0]]
            sy=[Dy[j] for j in np.nonzero(self.merge_dict==v)[0]]
            resultx.append(tf.add_n(sx))
            resulty.append(tf.add_n(sy))
        resultx=tf.concat(resultx,-1)
        resulty=tf.concat(resulty,-1)
        result = tf.stack([tf.reduce_sum(resulty,[1,2])/(tf.count_nonzero(resulty,[1,2],dtype=tf.float32)+1e-6),
                           tf.reduce_sum(resultx,[1,2])/(1e-6+tf.count_nonzero(resultx,[1,2],dtype=tf.float32))],axis=-1)
        return result
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[3],2)



class MYSGD(keras.optimizers.SGD):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            if 'class_conv/kernel' in p.name:
                v=self.momentum * m - 10*lr * g
            elif 'class_conv/bias' in p.name:
                v=self.momentum * m - 20*lr * g
            else:
                v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
def build_maskconv_model():
    y=KL.Input([None,None,256],dtype=tf.float32)
    x=KL.Conv2D(256,(3,3),padding='same',name='mask_conv1')(y)
    x=KL.BatchNormalization(name='mask_bn1')(x)
    x=KL.Activation('relu')(x)
    x=KL.Conv2D(256,(3,3),padding='same',name='mask_conv2')(x)
    x=KL.BatchNormalization(name='mask_bn2')(x)
    x=KL.Activation('relu')(x)
    x=KL.Conv2D(256,(3,3),padding='same',name='mask_conv3')(x)
    x=KL.BatchNormalization(name='mask_bn3')(x)
    x=KL.Activation('relu')(x)
    x=KL.Conv2D(256,(3,3),padding='same',name='mask_conv4')(x)
    x=KL.BatchNormalization(name='mask_bn4')(x)
    x=KL.Activation('relu')(x)
    return KM.Model(inputs=[y], outputs=[x],name='maskconv_model')
class Resnet101FCN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir,trainmode='finetune'):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config,train_submode=trainmode)

    def build(self, mode, config,train_submode='finetune'):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h/2**5 != int(h/2**5) or w/2**5 != int(w/2**5):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w, 1], axis=0), tf.float32)
            # GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]

            input_gt_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],
                name="input_gt_masks", dtype=tf.uint8)

            gt_masks=KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.uint8),config.NUM_CLASSES),name='input_onehot')(input_gt_masks)


        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        mask_feature_maps = [P2, P3, P4, P5]
        mask_logits_outputs=[]
        Conv_model =build_maskconv_model()
        for i,v in enumerate( mask_feature_maps):
            x=Conv_model(v)
            x=KL.Lambda(lambda x:tf.image.resize_bilinear(x,config.BACKBONE_SHAPES[0]))(x)
            mask_logits_outputs.append(x)
        x=KL.average(mask_logits_outputs)
        x=KL.Conv2D(config.NUM_CLASSES,(3,3),padding='same',name='mask_class_conv')(x)
        mask_logits=KL.Lambda(lambda x:tf.image.resize_bilinear(x,[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]]),name='mask_logits')(x)
        mask_prob = KL.Activation('softmax',name='mask_prob')(mask_logits)
        mask=KL.Lambda(lambda x:tf.argmax(x,-1),name='mask')(mask_prob)
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            if train_submode=='finetune':
                mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x,OHEM=True), name="resfcn_mask_loss")(
                    [gt_masks,mask_prob])
            elif train_submode=='finetune_ssloss':
                generate_gmap=GRenderLayer(1,map_size=[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]],name='generate_gmap')
                generate_centers=ComputeCentersLayer(num_classes=config.NUM_CLASSES,name='generate_centers')
                predict_mask = KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.int32),config.NUM_CLASSES))(mask)
                c1=generate_centers(gt_masks)
                c2=generate_centers(predict_mask)
                gt_cmap=generate_gmap(c1)
                pre_cmap=generate_gmap(c2)
                mask_loss=KL.Lambda(lambda x:ss_mask_loss_graph(*x),name="resfcn_mask_loss")([gt_masks,mask_prob,gt_cmap,pre_cmap])
                outputs = [mask_prob,mask]
            else:
                print('Incorrect Option of TrainSubMode')
                raise FileNotFoundError
            # Model
            inputs = [input_image, input_image_meta, input_gt_masks]
            pixelacc =KL.Lambda(lambda x:pixelacc_graph(*x),name='pixelacc')([gt_masks,mask_prob])
            outputs = [mask_prob,mask,mask_loss,pixelacc]
            model = KM.Model(inputs, outputs, name='resnet101_fcn')
        else:
            model = KM.Model([input_image, input_image_meta],
                                    [mask_prob,mask],
                                    name='resnet101_fcn')
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.__class__.__name__), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            models = filter(lambda l: l.__class__.__name__ =='Model', layers)
            players = filter(lambda l: l.__class__.__name__ !='Model', layers)
            topology.load_weights_from_hdf5_group_by_name(f, players)
            for v in models:
                weights=[]
                for j in v.weights:
                    try:
                        weights.append((j,f[v.name][j.name]))
                    except:
                        print(j.name+' not found !\n')
                K.batch_set_value(weights)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = MYSGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["resfcn_mask_loss","pixelacc"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=False))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                    for w in self.keras_model.trainable_weights]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None]*len(self.keras_model.outputs))

        # Add metrics
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output,
                                                    keep_dims=False))

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent+4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/%s\_%s\_(\d{4})\.h5"%(self.__class__.__name__,self.config.NAME.lower())
            m = re.match(regex, model_path)
            if m:
                print('Matching !!!!\n')
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_{}_*epoch*.h5".format(self.__class__.__name__,
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")


    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "5+": r"(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            # All layers
            'psp5+':"(resfcn\_.*)|(fpn\_.*)|(mask.*)|(conv5.*)",
            'head':"(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                                batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                            batch_size=self.config.BATCH_SIZE,augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True,period=1),
            keras.callbacks.LearningRateScheduler(lambda x:learning_rate if x<epochs*0.6 else 0.1*learning_rate)
        ]

        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "validation_data":val_generator,
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": 2,
            "workers": 1,#max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": False,
        }

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils_resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            # Build image_meta
            molded_image=mold_image(molded_image,self.config)
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows
    def unmold_detections_filter(self, detections, mrcnn_mask, image_shape, window,superpixel_map):
            """Reformats the detections of one image from the format of the neural
            network output to a format suitable for use in the rest of the
            application.

            detections: [N, (y1, x1, y2, x2, class_id, score)]
            mrcnn_mask: [N, height, width, num_classes]
            image_shape: [height, width, depth] Original size of the image before resizing
            window: [y1, x1, y2, x2] Box in the image where the real image is
                    excluding the padding.

            Returns:
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
            """
            # How many detections do we have?
            # Detections array is padded with zeros. Find the first class_id == 0.
            zero_ix = np.where(detections[:,4] == 0)[0]
            N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

            # Extract boxes, class_ids, scores, and class-specific masks
            boxes = detections[:N, :4]
            class_ids = detections[:N, 4].astype(np.int32)
            scores = detections[:N, 5]
            masks = mrcnn_mask[np.arange(N), :, :, class_ids]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 2] - boxes[:, 0]) <= 0)[0]
            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)
                masks = np.delete(masks, exclude_ix, axis=0)
                N = class_ids.shape[0]

            # Compute scale and shift to translate coordinates to image domain.
            h_scale = image_shape[0] / (window[2] - window[0])
            w_scale = image_shape[1] / (window[3] - window[1])
            scale = min(h_scale, w_scale)
            shift = window[:2]  # y, x
            scales = np.array([scale, scale, scale, scale])
            shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

            # Translate bounding boxes to image domain
            boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

            # Resize masks to original image size and set boundary threshold.
            full_masks = []
            for i in range(N):
                # Convert neural network mask to full size mask
                full_mask = unmold_mask_superpixel_filter(masks[i], boxes[i], image_shape,superpixel_map)
                full_masks.append(full_mask)
            full_masks = np.stack(full_masks, axis=-1)\
                        if full_masks else np.empty((0,) + masks.shape[1:3])

            return boxes, class_ids, scores, full_masks
    def unmold_detections(self,mrcnn_mask_prob,mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        mask=mrcnn_mask[window[0]:window[2],window[1]:window[3]]
        mask=scipy.ndimage.zoom(mask,[h_scale,w_scale],order=0)
        mask_prob=mrcnn_mask_prob[window[0]:window[2],window[1]:window[3]]
        mask_prob=scipy.ndimage.zoom(mask_prob,[h_scale,w_scale,1],order=2)
        # Translate bounding boxes to image domain

        # Resize masks to original image size and set boundary threshold.

        return mask_prob,mask
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        mask_prob,mask =\
            self.keras_model.predict([molded_images, image_metas], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_mask_probs,final_masks =\
                self.unmold_detections(mask_prob[i],mask[i],
                                       image.shape, windows[i])
            results.append({
                "probs": final_mask_probs,
                "masks": final_masks,
            })
        return results
    def detect_filter(self, images,superpixel_map,verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
        rois, rpn_class, rpn_bbox =\
            self.keras_model.predict([molded_images, image_metas], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections_filter(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i],superpixel_map)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results
    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers
    def get_layers_by_name(self,names):
        layers =OrderedDict().fromkeys(names)
        for v in self.keras_model.layers:
            if v.name in names:
                layers[v.name]=v.output
        return layers
    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas,
        #                 target_rpn_match, target_rpn_bbox,
        #                 gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################
def build_attresnet_model(architecture,stage5=False):
    x=KL.Input([None,None,3])
    res = ResNet(x,layers=101)
    return KM.Model([x],[res],name=architecture+'_model')
def build_mutiscale_image(image,image_shape,scales=0.5):
    s=scales
    x = tf.image.resize_bilinear(image,tf.cast(tf.convert_to_tensor(image_shape[:2],dtype=tf.float32)*s,tf.int32))
    return x
def build_weighted_feature(inputs):
    z=[]
    feature_map = inputs[:-1]
    weights =tf.split(inputs[-1],len(feature_map),-1)
    for i in range(len(feature_map)):
        z.append(feature_map[i]*weights[i])
    return tf.add_n(z)
def ss_mask_loss_graph(target_masks,pred_masks,target_cmap,pred_cmap,OHEM=True,KK=3):
    loss =K.categorical_crossentropy(target_masks,pred_masks)*K.sum(target_masks,-1)
    ss_loss = K.mean(K.sum((target_cmap-pred_cmap)**2,[1,2]),axis=-1)
    if OHEM:
        shape = tf.shape(loss)
        loss,_ = tf.nn.top_k(tf.reshape(loss,[-1,shape[1]*shape[2]]),k=tf.cast(shape[1]*shape[2]/KK,tf.int32),sorted=False)
        loss = tf.reduce_mean(loss,axis=-1)
    else:
        loss = K.sum(loss,[1,2])/K.sum(target_masks,[1,2,3])
    loss = K.mean(0.5*ss_loss*loss)
    loss = K.reshape(loss, [1, 1])
    return loss
def pixelacc_graph(gt_masks,pred_masks):
    x=keras.metrics.categorical_accuracy(gt_masks,pred_masks)*K.sum(gt_masks,-1)
    x=K.mean(K.sum(x,[1,2])/K.sum(gt_masks,[1,2,3]))
    x=K.reshape(x,[1,1])
    return x
class AttResnet101FCN(Resnet101FCN):
    def __init__(self, mode, config, model_dir,trainmode='finetune'):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config,train_submode=trainmode)
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = MYSGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ['att_mask_loss','pixelacc']
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=False))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                    for w in self.keras_model.trainable_weights]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None]*len(self.keras_model.outputs))

        # Add metrics
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output,
                                                    keep_dims=False))

    def build(self, mode, config,train_submode='finetune'):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]


        # Inputs
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w, 1], axis=0), tf.float32)
            # GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]

            input_gt_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],
                name="input_gt_masks", dtype=tf.uint8)

            gt_masks=KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.uint8),config.NUM_CLASSES),name='input_onehot')(input_gt_masks)


        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet= build_attresnet_model("resnet101", stage5=True)
        mutiscal_images=[]
        mutiscal_feature=[]
        ResizeLayer=KL.Lambda(lambda x:tf.image.resize_bilinear(x,config.BACKBONE_SHAPES[1]))
        if mode =='training' and train_submode=='pretrain':
            s=1-0.25*tf.cast(K.random_uniform(shape=(),minval=0,maxval=3,dtype=tf.int32),tf.float32)
            for t in range(3):
                mutiscal_images.append(KL.Lambda(lambda x:build_mutiscale_image(x,[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],scales=s))(input_image))
            for v in mutiscal_images:
                x = resnet(v)
                x=ResizeLayer(x)
                mutiscal_feature.append(x)
        else:
            for s in [0.5,0.75,1.0]:
                mutiscal_images.append(KL.Lambda(lambda x:build_mutiscale_image(x,[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],scales=s))(input_image))
            for v in mutiscal_images:
                x = resnet(v)
                x=ResizeLayer(x)
                mutiscal_feature.append(x)

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        x =KL.concatenate(mutiscal_feature)
        Att =KL.Conv2D(512,[3,3],name='mask_att_conv',activation='relu',padding='same')(x)
        Att=KL.Dropout(0.5)(Att)
        Att=KL.Conv2D(3,[3,3],activation='softmax',name='mask_att_prob',padding='same')(Att)
        weighted_features=KL.Lambda(lambda x:build_weighted_feature(x),name='weigthed_features')(mutiscal_feature+[Att])
        dense_classifier = KL.Conv2D(config.NUM_CLASSES,(3,3),name='mask_class_conv',padding='same')
        finetune_dense_class=KL.Conv2D(config.NUM_CLASSES,(3,3),name='mask_denseclass_conv',padding='same')
        x=finetune_dense_class(weighted_features)
        #x=dense_classifier(weighted_features)
        mask_generate=KL.Lambda(lambda x:tf.image.resize_bilinear(x,[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]]),name='mask_logits')

        mask_logits=mask_generate(x)
        prob_generate = KL.Activation('softmax',name='mask_prob')
        mask_prob = prob_generate(mask_logits)
        label_generate =KL.Lambda(lambda x:tf.argmax(x,-1,output_type=tf.int32),name='mask')
        mask=label_generate(mask_prob)
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            deeply_supervise= [prob_generate(mask_generate(dense_classifier(v))) for v in mutiscal_feature]
            mask_loss_compute = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="resfcn_mask_loss")
            outputs=[]
            if 'pretrain' in train_submode:
                s_prob=deeply_supervise[0]
                outputs=[s_prob]
                mask_loss =KL.Lambda(lambda x:x,name='att_mask_loss')(mask_loss_compute([gt_masks,s_prob]))
            elif train_submode=='finetune':
                mask_loss=KL.Lambda(lambda x:x,name='att_mask_loss')(mask_loss_compute([gt_masks,mask_prob]))
                outputs=[mask_prob,mask]
            elif train_submode=='finetune_withdeep':
                mask_loss = KL.add([mask_loss_compute([gt_masks,v]) for v in deeply_supervise+[mask_prob]],name='att_mask_loss')
            elif train_submode=='finetune_ssloss':
                generate_gmap=GRenderLayer(1,map_size=[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]],name='generate_gmap')
                generate_centers=ComputeCentersLayer(num_classes=config.NUM_CLASSES,name='generate_centers')
                predict_mask = KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.int32),config.NUM_CLASSES))(mask)
                c1=generate_centers(gt_masks)
                c2=generate_centers(predict_mask)
                gt_cmap=generate_gmap(c1)
                pre_cmap=generate_gmap(c2)
                mask_loss=KL.Lambda(lambda x:ss_mask_loss_graph(*x),name='att_mask_loss')([gt_masks,mask_prob,gt_cmap,pre_cmap])
                outputs = [mask_prob,mask]
            elif train_submode=='finetune_ssloss_withdeep':
                deeply_supervise= [prob_generate(mask_generate(finetune_dense_class(v))) for v in mutiscal_feature]
                label_masks =[label_generate(v) for v in deeply_supervise]+[mask]
                generate_gmap=GRenderLayer(1,map_size=[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]],name='generate_gmap')
                generate_centers=ComputeCentersLayer(num_classes=config.NUM_CLASSES,name='generate_centers')
                predict_masks = [KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.int32),config.NUM_CLASSES))(v) for v in label_masks]
                gt_cmap=generate_gmap(generate_centers(gt_masks))
                pre_centers=[generate_centers(v) for v in predict_masks]
                pre_cmaps=[generate_gmap(v) for v in pre_centers]
                mask_probs =deeply_supervise+[mask_prob]
                ss_loss = KL.Lambda(lambda x:ss_mask_loss_graph(*x))
                mask_loss=KL.average([ss_loss([gt_masks,mask_probs[i],gt_cmap,pre_cmaps[i]]) for i in range(len(
                                                                                                       pre_cmaps))],name='att_mask_loss')
                outputs = [mask_prob,mask]
            else:
                print('Incorrect Option of TrainSubMode')
                raise FileNotFoundError
            # Model
            inputs = [input_image, input_image_meta, input_gt_masks]
            pixelacc =KL.Lambda(lambda x:pixelacc_graph(*x),name='pixelacc')([gt_masks,outputs[0]])
            outputs = outputs+[mask_loss,pixelacc]
            model = KM.Model(inputs, outputs, name='resnet101_fcn')
        else:
            model = KM.Model([input_image, input_image_meta],
                                    [mask_prob,mask],
                                    name='resnet101_fcn')
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "5+": r"(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            # All layers
            'psp5+':"(resfcn\_.*)|(fpn\_.*)|(mask.*)|(conv5.*)",
            'head':"(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                                batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                            batch_size=self.config.BATCH_SIZE,augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True,period=1),
            keras.callbacks.LearningRateScheduler(lambda x:learning_rate*(1-x/epochs)**0.9)
        ]

        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "validation_data":val_generator,
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": 2,
            "workers": 1,#max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": False,
        }

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )
        self.epoch = max(self.epoch, epochs)
def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.

    TODO: use this function to reduce code duplication
    """
    area = tf.boolean_mask(boxes, tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1),
                           tf.bool))


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)






def build_attresize_resnet_model(architecture='resnet101'):
    x=KL.Input([None,None,3])
    C1,C2,C3,C4,C5 = ResNet_mutiout(x,layers=101)
    return KM.Model([x],[C1,C2,C3,C4,C5],name=architecture+'_model')

def compute_attention(x,y,z,dk=256.0,vk=2048):
    x_shape= tf.shape(x)
    y_shape= tf.shape(y)
    z_shape = tf.shape(z)
    att =tf.matmul(tf.reshape(x,[x_shape[0],-1,x_shape[-1]]),tf.reshape(y,[y_shape[0],-1,y_shape[-1]]),transpose_b=True)/tf.sqrt(1.0*dk)
    att = tf.nn.softmax(att)
    value = tf.matmul(att,tf.reshape(z,[z_shape[0],-1,z_shape[-1]]))
    value =tf.reshape(value,[x_shape[0],x_shape[1],x_shape[2],vk])
    return value
def build_softatt_block(Q,K,V,embedding_channel_size=256,value_channel_size=2048,name='attblock'):
    Q_emb =KL.Conv2D(embedding_channel_size,(1,1),padding='same',name=name+'_qembconv')(Q)
    K_emb =KL.Conv2D(embedding_channel_size,(1,1),padding='same',name=name+'_vembconv')(K)
    values = KL.Lambda(lambda x:compute_attention(*x, dk=embedding_channel_size,vk=value_channel_size),name =name+'_value')([Q_emb,K_emb,V])
    return values





class AttUpsampleFCN(Resnet101FCN):
    def __init__(self, mode, config, model_dir,trainmode='finetune'):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config,train_submode=trainmode)
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = MYSGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ['att_mask_loss','pixelacc']
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=False))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                    for w in self.keras_model.trainable_weights]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None]*len(self.keras_model.outputs))

        # Add metrics
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output,
                                                    keep_dims=False))

    def build(self, mode, config,train_submode='finetune'):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]


        # Inputs
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w, 1], axis=0), tf.float32)
            # GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]

            input_gt_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],
                name="input_gt_masks", dtype=tf.uint8)

            gt_masks=KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.uint8),config.NUM_CLASSES),name='input_onehot')(input_gt_masks)


        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet= build_attresize_resnet_model("resnet101")
        ResizeLayer=KL.Lambda(lambda x:tf.image.resize_bilinear(x,config.BACKBONE_SHAPES[1]))
        C1,C2,C3,C4,C5 = resnet(input_image)
        # Attention Upsampling
        s1 = build_softatt_block(C4,C4,C5,512,2048,name='mask_attup1')
        s2 = build_softatt_block(C3,C3,C5,256,2048,name='mask_attup2')
        s4 = build_softatt_block(ResizeLayer(C1),ResizeLayer(C1),C5,128,2048,name='mask_attup4')
        dense_classifier = KL.Conv2D(config.NUM_CLASSES,(3,3),name='mask_class_conv',padding='same')
        aver_feature = KL.average([C5,s1,s2,s4],name='mask_aver_feature')
        x=dense_classifier(aver_feature)
        mask_generate=KL.Lambda(lambda x:tf.image.resize_bilinear(x,[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]]),name='mask_logits')
        mask_logits=mask_generate(x)
        prob_generate = KL.Activation('softmax',name='mask_prob')
        mask_prob = prob_generate(mask_logits)
        label_generate =KL.Lambda(lambda x:tf.argmax(x,-1,output_type=tf.int32),name='mask')
        mask=label_generate(mask_prob)
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            mask_loss_compute = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="resfcn_mask_loss")
            if train_submode=='finetune':
                mask_loss=KL.Lambda(lambda x:x,name='att_mask_loss')(mask_loss_compute([gt_masks,mask_prob]))
                outputs=[mask_prob,mask]
            elif train_submode=='finetune_ssloss':
                generate_gmap=GRenderLayer(1,map_size=[config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]],name='generate_gmap')
                generate_centers=ComputeCentersLayer(num_classes=config.NUM_CLASSES,name='generate_centers')
                predict_mask = KL.Lambda(lambda x:K.one_hot(tf.cast(x,tf.int32),config.NUM_CLASSES))(mask)
                c1=generate_centers(gt_masks)
                c2=generate_centers(predict_mask)
                gt_cmap=generate_gmap(c1)
                pre_cmap=generate_gmap(c2)
                mask_loss=KL.Lambda(lambda x:ss_mask_loss_graph(*x),name='att_mask_loss')([gt_masks,mask_prob,gt_cmap,pre_cmap])
                outputs = [mask_prob,mask]
            else:
                print('Incorrect Option of TrainSubMode')
                raise FileNotFoundError
            # Model
            inputs = [input_image, input_image_meta, input_gt_masks]
            pixelacc =KL.Lambda(lambda x:pixelacc_graph(*x),name='pixelacc')([gt_masks,outputs[0]])
            outputs = outputs+[mask_loss,pixelacc]
            model = KM.Model(inputs, outputs, name='resnet101_fcn')
        else:
            model = KM.Model([input_image, input_image_meta],
                                    [mask_prob,mask],
                                    name='resnet101_fcn')
        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "5+": r"(res5.*)|(bn5.*)|(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            # All layers
            'psp5+':"(resfcn\_.*)|(fpn\_.*)|(mask.*)|(conv5.*)",
            'head':"(resfcn\_.*)|(fpn\_.*)|(mask.*)",
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                                batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                            batch_size=self.config.BATCH_SIZE,augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True,period=1),
            keras.callbacks.LearningRateScheduler(lambda x:learning_rate*(1-x/epochs)**0.9)
        ]

        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            "validation_data":val_generator,
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": 2,
            "workers": 1,#max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": False,
        }

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
            )
        self.epoch = max(self.epoch, epochs)
