import requests
import pickle
import cv2
import numpy as np
import time
import logging
import tensorflow as tf
from model_zalando_mask_content import create_model
import scipy.io as sio
from utils import (extract_pose_keypoints,
                   extract_pose_map,
                   extract_segmentation,
                   process_segment_map)
import SS_NAN.visualize as visualize
from PIL import Image
import pdb

LOGGING_LEVEL = logging.INFO
logging.basicConfig(
        level=LOGGING_LEVEL,
        format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
                '%(levelname)s - %(message)s'),
        )
logger = logging.getLogger(__name__)

####################
# VITON Parameters #
####################
POSE_NORM_PATH = 'data/pose/000001_0.mat'
SEG_NORM_PATH = 'data/segment/000001_0.mat'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('b', '', 'Server address')
tf.app.flags.DEFINE_integer('w', 1, 'Number of workers')
tf.app.flags.DEFINE_integer('timeout', 120, 'Server timeout')
tf.logging.set_verbosity(tf.logging.INFO)

cap = cv2.VideoCapture(0)


class VITONDemo():
    def __init__(self):
        logger.info("Loading VITON_worker ...")
        self.batch_size = 1
        self.image_holder = \
            tf.placeholder(tf.float32,
                           shape=[self.batch_size, 256, 192, 3])
        self.prod_image_holder = tf.placeholder(
                tf.float32, shape=[self.batch_size, 256, 192, 3])
        self.body_segment_holder = tf.placeholder(
                tf.float32, shape=[self.batch_size, 256, 192, 1])
        self.prod_segment_holder = tf.placeholder(
                tf.float32, shape=[self.batch_size, 256, 192, 1])
        self.skin_segment_holder = tf.placeholder(
                tf.float32, shape=[self.batch_size, 256, 192, 3])
        self.pose_map_holder = \
            tf.placeholder(tf.float32,
                           shape=[self.batch_size, 256, 192, 18])
        self.viton_worker = create_model(self.prod_image_holder,
                                         self.body_segment_holder,
                                         self.skin_segment_holder,
                                         self.pose_map_holder,
                                         self.prod_segment_holder,
                                         self.image_holder)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        logger.info("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        if checkpoint is None:
            checkpoint = FLAGS.checkpoint
        logger.info("Checkpoint: {}".format(checkpoint))
        saver.restore(self.sess, checkpoint)

        logger.info("Initialization done")

    def _process_image(self, image, prod_image,
                       pose_raw, segment_raw,  sess,
                       resize_width=192, resize_height=256):
        if len(pose_raw) == 0:
            logger.warning("No pose")
            pose_raw = sio.loadmat(POSE_NORM_PATH)
            pose_raw = extract_pose_keypoints(pose_raw)
            pose_raw = extract_pose_map(pose_raw,
                                        image.shape[0],
                                        image.shape[1])
            pose_raw = np.asarray(pose_raw, np.float32)
        else:
            pose_tmp = []
            for i in range(18):
                if i in pose_raw:
                    pose_tmp.append(pose_raw[i])
                else:
                    pose_tmp.append([-1, -1])
            pose_raw = np.array(pose_tmp)
            pose_raw = extract_pose_map(pose_raw,
                                        image.shape[0],
                                        image.shape[1])
            pose_raw = np.asarray(pose_raw, np.float32)
        assert pose_raw.shape == (256, 192, 18)

        if len(segment_raw) == 0:
            logger.warning("No seg")
            segment_raw = sio.loadmat(SEG_NORM_PATH)["segment"]
            segment_raw = process_segment_map(segment_raw,
                                              image.shape[0],
                                              image.shape[1])
        else:
            segment_raw = np.asarray(segment_raw, np.uint8)
        segment_deb = sio.loadmat(SEG_NORM_PATH)["segment"]
        segment_deb = process_segment_map(segment_deb,
                                          image.shape[0],
                                          image.shape[1])
        # segment_raw = segment_deb

        (body_segment, prod_segment, skin_segment) = (
                extract_segmentation(segment_raw))

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)

        image = tf.image.resize_images(image,
                                       size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)
        prod_image = \
            tf.image.resize_images(prod_image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

        body_segment = \
            tf.image.resize_images(body_segment,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   align_corners=False)
        skin_segment = \
            tf.image.resize_images(skin_segment,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   align_corners=False)

        prod_segment = \
            tf.image.resize_images(prod_segment,
                                   size=[resize_height, resize_width],
                                   method=(tf.image
                                           .ResizeMethod.NEAREST_NEIGHBOR))

        image = (image - 0.5) * 2.0
        prod_image = (prod_image - 0.5) * 2.0

        # using skin rbg
        skin_segment = skin_segment * image

        [image, prod_image, body_segment, prod_segment, skin_segment] = \
            sess.run([image, prod_image, body_segment,
                      prod_segment, skin_segment])
        # pdb.set_trace()

        return (image, prod_image, pose_raw,
                body_segment, prod_segment, skin_segment)

    def viton_infer(self, frame, product_image, pose_raw, segment_raw):
        images = np.zeros((self.batch_size, 256, 192, 3))
        prod_images = np.zeros((self.batch_size, 256, 192, 3))
        body_segments = np.zeros((self.batch_size, 256, 192, 1))
        prod_segments = np.zeros((self.batch_size, 256, 192, 1))
        skin_segments = np.zeros((self.batch_size, 256, 192, 3))
        pose_raws = np.zeros((self.batch_size, 256, 192, 18))
        for i in range(self.batch_size):
            (image, prod_image, pose_raw,
             body_segment, prod_segment,
             skin_segment) = self._process_image(frame,
                                                 product_image,
                                                 pose_raw,
                                                 segment_raw,
                                                 self.sess)
            images[i] = image
            prod_images[i] = prod_image
            body_segments[i] = body_segment
            prod_segments[i] = prod_segment
            skin_segments[i] = skin_segment
            pose_raws[i] = pose_raw
        feed_dict = {
                self.image_holder: images,
                self.prod_image_holder: prod_images,
                self.body_segment_holder: body_segments,
                self.skin_segment_holder: skin_segments,
                self.prod_segment_holder: prod_segments,
                self.pose_map_holder: pose_raws,
        }
        [image_output, mask_output, loss, step] = self.sess.run(
                [self.viton_worker.image_outputs,
                 self.viton_worker.mask_outputs,
                 self.viton_worker.gen_loss_content_L1,
                 self.viton_worker.global_step],
                feed_dict=feed_dict)
        return image_output[0]


def draw_segment_mask(frame, masks):
    color = visualize.random_colors(N=20)
    frame = visualize.apply_mask(frame, masks, color=color,
                                 class_ids=[v for v in
                                            range(1,
                                                  20)])
    return frame


demo = VITONDemo()

while 1:
    t = time.time()
    ret, img = cap.read()
    img_name = 'data/women_top/000001_0.jpg'
    img_name = 'test_person2.jpg'
    img_name = 'test_person.jpg'
    prod_name = 'data/women_top/001744_1.jpg'
    # img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_name = 'tmp.jpg'
    cv2.imwrite(img_name, img)

    # Get pose
    logger.info("Getting pose ..")
    files = {'files': open(img_name, 'rb')}
    url = 'http://140.112.29.182:8000/pose'
    r = requests.post(url, files=files)
    poses = pickle.loads(r.content)

    # Get seg masks
    logger.info("Getting seg ..")
    files = {'files': open(img_name, 'rb')}
    url = 'http://140.112.29.182:8000/seg'
    r = requests.post(url, files=files)
    masks = pickle.loads(r.content)
    masked_img = draw_segment_mask(img.copy(), masks)
    cv2.imshow('a', masked_img)

    prod_img = np.array(Image.open(prod_name))
    output = demo.viton_infer(img, prod_img, poses, masks)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = output / 2.0 + 0.5
    cv2.imshow('b', output)
    print(1 / (time.time() - t))
    k = cv2.waitKey(100)
    if k == ord('q'):
        break
