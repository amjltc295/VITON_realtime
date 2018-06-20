from flask import Flask, request
from PIL import Image
import pickle
import numpy as np
import cv2
import logging
import os
import tensorflow as tf
import scipy.io as sio
from model_zalando_mask_content import create_model
from utils import (extract_pose_keypoints,
                   extract_pose_map,
                   extract_segmentation,
                   process_segment_map)
from flat_attach_clothes import flat_attach_clothes
# import pdb

LOGGING_LEVEL = logging.INFO
logging.basicConfig(
        level=LOGGING_LEVEL,
        format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
                '%(levelname)s - %(message)s'),
        )
logger = logging.getLogger(__name__)
(VIDEO_CH, VIDEO_H, VIDEO_W) = (3, 480, 640)


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


class VITONDemo():
    def __init__(self):
        self.prod_name = './inputs/a.jpg'
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

    def set_prod_name(self, name):
        name = './inputs/' + str(name)
        if os.path.exists(name):
            self.prod_name = name
            return True
        return False

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


demo = VITONDemo()
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "Welcome"


@app.route("/viton", methods=["GET"])
def viton():
    logger.info("VITON inferring ...")
    pose_and_seg_data = pickle.load(open('pose_and_seg_data.pickle', 'rb'))
    poses = pose_and_seg_data['poses']
    masks = pose_and_seg_data['masks']
    img = pose_and_seg_data['frame']
    prod_img = np.array(Image.open(demo.prod_name))
    output = demo.viton_infer(img, prod_img, poses, masks)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = cv2.resize(output, (img.shape[1], img.shape[0]))
    output = output / 2.0 + 0.5
    output *= 255
    cv2.imwrite('tmp_out.jpg', output)

    return "Done"


@app.route("/attach", methods=["GET"])
def attach():
    logger.info("Attatch ..")
    pose_and_seg_data = pickle.load(open('pose_and_seg_data.pickle', 'rb'))
    masks = pose_and_seg_data['masks']
    img = pose_and_seg_data['frame']
    prod_img = np.array(Image.open(demo.prod_name))
    attached_img = flat_attach_clothes(masks, img, prod_img)
    attached_img = cv2.cvtColor(attached_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('tmp_out2.jpg', attached_img)
    return "Done"


@app.route("/change", methods=["POST"])
def change():
    name = request.data.decode("utf-8")
    if not demo.set_prod_name(name):
        msg = "No cloth name {}".format(name)
    else:
        msg = "Cloth changed to {}".format(name)
    logger.info(msg)
    return msg
