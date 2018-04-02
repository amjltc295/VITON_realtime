from flask import Flask, request
from PIL import Image
import pickle
import numpy as np
import cv2
import logging
import tensorflow as tf
from tf_pose_estimation.src.networks import get_graph_path
from tf_pose_estimation.src.estimator import TfPoseEstimator as Pose_Inferrer
from tf_pose_estimation.src.common import (CocoPart,
                                           CocoColors, CocoPairsRender)
import SS_NAN.LIP
import SS_NAN.visualize as visualize
from SS_NAN.model import AttResnet101FCN as Seg_Inferrer
from model_zalando_mask_content import create_model
from utils import extract_segmentation
import pdb

LOGGING_LEVEL = logging.INFO
logging.basicConfig(
        level=LOGGING_LEVEL,
        format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
                '%(levelname)s - %(message)s'),
        )
logger = logging.getLogger(__name__)
(VIDEO_CH, VIDEO_H, VIDEO_W) = (3, 480, 640)

#####################
# SS_NAN Parameters #
#####################
SS_NAN_MODEL_DIR = './SS_NAN/model/logs'
LIP_MODEL_PATH = './SS_NAN/AttResnet101FCN_lip_0023.h5'

####################
# VITON Parameters #
####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('b', '', 'Server address')
tf.app.flags.DEFINE_integer('w', 1, 'Number of workers')
tf.app.flags.DEFINE_integer('timeout', 120, 'Server timeout')
tf.logging.set_verbosity(tf.logging.INFO)


class SegInferenceConfig(SS_NAN.LIP.LIPConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MAX_DIM = 640


seg_config = SegInferenceConfig()
seg_config.display()


def draw_humans(npimg, humans, imgcopy=False, return_frame=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5),
                      int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i],
                       thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() \
                    or pair[1] not in human.body_parts.keys():
                continue

            npimg = cv2.line(npimg, centers[pair[0]],
                             centers[pair[1]], CocoColors[pair_order], 3)
    if return_frame:
        return npimg
    else:
        return centers


class VITONDemo():
    def __init__(self):
        logger.info("Loading pose_inferrer ...")
        self.pose_inferrer = Pose_Inferrer(get_graph_path('mobilenet_thin'),
                                           target_size=(VIDEO_W, VIDEO_H))
        logger.info("Creating seg_inferrer ...")
        self.seg_inferrer = Seg_Inferrer(mode="inference",
                                         model_dir=SS_NAN_MODEL_DIR,
                                         config=seg_config)
        logger.info("Loading seg_inferrer weight...")
        self.seg_inferrer.load_weights(LIP_MODEL_PATH, by_name=True)
        self.seg_inferrer.keras_model._make_predict_function()
        """
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
        with self.sess:
            logger.info("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
            if checkpoint is None:
                checkpoint = FLAGS.checkpoint
            logger.info("Checkpoint: {}".format(checkpoint))
            saver.restore(self.sess, checkpoint)
        """

        logger.info("Initialization done")

    def _process_image(self, image, prod_image,
                       pose_raw, segment_raw,  sess,
                       resize_width=192, resize_height=256):
        pdb.set_trace()
        pose_raw = np.asarray(pose_raw, np.float32)

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

        return (image, prod_image, pose_raw,
                body_segment, prod_segment, skin_segment)

    def pose_infer(self, frame):
        humans = self.pose_inferrer.inference(frame)
        frame = draw_humans(frame, humans, imgcopy=False)
        return frame

    def seg_infer(self, frame):
        results = self.seg_inferrer.detect([frame], verbose=0)
        masks = results[0]['masks']
        return masks
        color = visualize.random_colors(N=seg_config.NUM_CLASSES)
        frame = visualize.apply_mask(frame, masks, color=color,
                                     class_ids=[v for v in
                                                range(1,
                                                      seg_config.NUM_CLASSES)])
        return frame

    def viton_infer(self, frame, product_image, pose_raw, segment_raw):
        pdb.set_trace()
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


@app.route("/pose", methods=["POST"])
def pose():
    logger.info("Pose inferreing ..")
    img = Image.open(request.files['files'])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    humans = demo.pose_infer(img)
    """
    import pdb
    pdb.set_trace()
    """
    humans_pickle = pickle.dumps(humans)
    logger.info("Pose inferreing Done")

    return humans_pickle


@app.route("/seg", methods=["POST"])
def segment():
    logger.info("Seg inferreing ..")
    img = Image.open(request.files['files'])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = demo.seg_infer(img)
    masks_pickle = pickle.dumps(masks)

    return masks_pickle


@app.route("/viton", methods=["POST"])
def viton():
    img = Image.open(request.files['files'])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    poses = demo.pose_infer(img)
    masks = demo.seg_infer(img)
    prod_img = np.array(Image.open('test/product.jpg'))
    output = demo.viton_infer(img, prod_img, poses, masks)
    logger.info("Seg inferreing Done")

    return output
