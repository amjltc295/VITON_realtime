""" Provides class for real-time VITON demo. """
import requests
import pickle
import cv2
import numpy as np
import time
import logging
import tensorflow as tf
# from model_zalando_mask_content import create_model
from tf_pose_estimation.src.common import (CocoColors, CocoPairsRender)
import scipy.io as sio
from utils import (extract_pose_keypoints,
                   extract_pose_map,
                   extract_segmentation,
                   process_segment_map)
import SS_NAN.visualize as visualize
# from PIL import Image
# import pdb
import os
from time import gmtime, strftime, sleep
import threading
from queue import Queue, Empty
from DigCtrl import digCtrl, arduinoInit


__author__ = "Ya-Liang Chang (Allen)"
__copyright__ = ("All rights reserved.")
__credits__ = ["Ya-Liang Chang (Allen)"]
__version__ = "0.1"
__license__ = "All rights reserved."
__maintainer__ = "Ya-Liang Chang (Allen)"
__email__ = "b03901014@ntu.edu.tw"
__status__ = "Development"


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
tf.logging.set_verbosity(tf.logging.INFO)

####################
# Windows Settings #
####################
OUT_WINDOW_NAME = 'VITON'
SEG_WINDOW_NAME = 'Segmentation'
POSE_WINDOW_NAME = 'Pose'
ORIGIN_WINDOW_NAME = 'input'
OUT2_WINDOW_NAME = 'Attached'
CLOTHES_WINDOW_NAME = 'Clothes'

##################
# Queue Settings #
##################
QUEUE_WAIT_TIME = 10
FRAME_QUEUE_MAX_SIZE = 5

###################
# Output Settings #
###################
TMP_IMAGE_NAME = 'tmp.jpg'
TMP_OUTPUT_NAME = 'tmp_out.jpg'
TMP_OUTPUT_NAME2 = 'tmp_out2.jpg'
TMP_DATA_PICKLE_NAME = 'pose_and_seg_data.pickle'
VIDEO_SOURCE = 1
VIDEO_SOURCE = './inputs/test2.mp4'
RECORD_VIDEO = True if VIDEO_SOURCE in [0, 1] else True
RECORD_RESULT_VIDEO = True
RECORD_IMAGES = True
VITON_OUTPUT_DIR = 'outputs'
if not os.path.exists(VITON_OUTPUT_DIR):
    os.makedirs(VITON_OUTPUT_DIR)

###################
# Server Settings #
###################
SEG_SERVER = os.environ.get('SEG_SERVER')
SEG_URL = SEG_SERVER + '/seg'
POSE_SERVER = os.environ.get('POSE_SERVER')
POSE_URL = POSE_SERVER + '/pose'
VITON_SERVER = os.environ.get('VITON_SERVER')
VITON_URL = VITON_SERVER + '/viton'
CHANGE_CLOTH_URL = VITON_SERVER + '/change'
ATTACH_URL = VITON_SERVER + '/attach'

####################
# Control Settings #
####################
MOTOR_CONTROL = False


class FrameReader(threading.Thread):
    """ Thread to read frame from webcam. """

    def __init__(self, cap, frame_queue, process_frame_queue, threadManager):
        super(FrameReader, self).__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.process_frame_queue = process_frame_queue
        self.threadManager = threadManager
        self.run_flag = True
        self.pause_flag = False
        self.put_process_flag = False

    def run(self):
        count = 0
        while self.run_flag:
            count += 1
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 0)
            if VIDEO_SOURCE in [0, 1]:
                frame = np.rot90(frame, 1)
            else:
                sleep(0.5)
            frame = cv2.resize(frame, (360, 480))
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)
            if self.put_process_flag:
                cv2.imwrite(TMP_IMAGE_NAME, frame)
                sleep(0.1)
                logger.info('Tmp image written')
                self.process_frame_queue.put(frame)
                self.put_process_flag = False

    def put_process(self):
        self.put_process_flag = True

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class SegmentationExtractor(threading.Thread):
    """ Thread to do instance segmentaion. """

    def __init__(self, process_frame_queue, segmentation_data_queue,
                 segment_frmae_queue,
                 batch_size):
        super(SegmentationExtractor, self).__init__()
        self.process_frame_queue = process_frame_queue
        self.segmentation_data_queue = segmentation_data_queue
        self.segment_frmae_queue = segment_frmae_queue
        self.run_flag = True
        self.pause_flag = False
        self.batch_size = batch_size
        """
        self.class_names =  ['BG', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                             'Upper-clothes', 'Dress', 'Coats', 'Socks',
                             'Pants','Jumpsuits',  'Scarf', 'Skirt',
                             'Face', 'Left-arm','Right-arm', 'Left-leg',
                             'Right-leg', 'Left-shoe', 'Right-shoe']
        """
        self.color = visualize.random_colors(N=20)

    def draw_segment_mask(self, frame, masks):
        frame = visualize.apply_mask(frame, masks, color=self.color,
                                     class_ids=[v for v in range(1, 20)])
        return frame

    def run(self):
        while self.run_flag:
            if self.pause_flag:
                time.sleep(0.1)
                continue
            start_time = time.time()
            frames = []
            for i in range(self.batch_size):
                try:
                    frame = self.process_frame_queue.get_nowait()
                except Empty:
                    # logger.warning("segExtractor not getting frames")
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            if frames == []:
                continue
            else:
                logger.info("Start process")

            logger.debug("Seg-pre: {} s".format(time.time() - start_time))
            start_time = time.time()
            try:
                logger.info("Getting seg")
                files = {'files': open(TMP_IMAGE_NAME, 'rb')}
                url = SEG_URL
                r = requests.post(url, files=files)
                masks = pickle.loads(r.content)
                masked_img = self.draw_segment_mask(frame.copy(), masks)
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
                self.segment_frmae_queue.put(masked_img)
                logger.info("Seg got")

            except AttributeError as err:
                logger.error("Error in iseg inferring: {}".format(err))

            segmentation_data = {'frame': frame,
                                 'masks': masks}
            if self.segmentation_data_queue.full():
                self.segmentation_data_queue.get()
            self.segmentation_data_queue.put(segmentation_data)
            logger.warning("Seg: {} s".format(time.time() - start_time))

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class PoseEstimator(threading.Thread):
    """ Thread to do pose estimation. """

    def __init__(self, segmentation_data_queue, pose_and_seg_data_queue,
                 pose_frame_queue):
        super(PoseEstimator, self).__init__()
        self.segmentation_data_queue = segmentation_data_queue
        self.pose_and_seg_data_queue = pose_and_seg_data_queue
        self.pose_frame_queue = pose_frame_queue
        self.run_flag = True
        self.pause_flag = False

    def draw_humans(self, npimg, centers, imgcopy=True):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        for i, center in centers.items():
            cv2.circle(npimg, center, 3, CocoColors[i],
                       thickness=3, lineType=8, shift=0)

            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in centers \
                        or pair[1] not in centers:
                    continue

                npimg = cv2.line(npimg, centers[pair[0]],
                                 centers[pair[1]], CocoColors[pair_order], 3)
        return npimg

    def run(self):
        while self.run_flag:
            if self.pause_flag:
                time.sleep(0.1)
                continue
            try:
                segmentation_data = (self.segmentation_data_queue
                                     .get(timeout=QUEUE_WAIT_TIME))
            except Empty:
                # logger.warning("PoseEstimator not getting frames")
                continue
            start_time = time.time()
            if 'masks' not in segmentation_data:
                self.pose_and_seg_data_queue.put(segmentation_data)
                logger.debug("Pose infer: {} s"
                             .format(time.time() - start_time))
                continue

            frame = segmentation_data['frame']

            # Get pose
            logger.info("Getting pose ..")
            files = {'files': open(TMP_IMAGE_NAME, 'rb')}
            url = POSE_URL
            r = requests.post(url, files=files)
            poses = pickle.loads(r.content)
            posed_img = self.draw_humans(frame, poses)
            posed_img = cv2.cvtColor(posed_img, cv2.COLOR_RGB2BGR)
            self.pose_frame_queue.put(posed_img)

            segmentation_data['poses'] = poses
            segmentation_data['frame'] = frame

            self.pose_and_seg_data_queue.put(segmentation_data)
            logger.warning("Pose infer: {} s".format(time.time() - start_time))

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class VITONWorker(threading.Thread):
    """ Thread to do id-matching between this frame and last frame. """

    def __init__(self, pose_and_seg_data_queue,
                 viton_frame_queue,
                 viton_frame_queue2):
        super(VITONWorker, self).__init__()
        self.pose_and_seg_data_queue = pose_and_seg_data_queue
        self.viton_frame_queue = viton_frame_queue
        self.viton_frame_queue2 = viton_frame_queue2
        self.run_flag = True
        self.pause_flag = False
        self.prod_name = './inputs/a.jpg'
        if RECORD_VIDEO and False:
            current_time = strftime("%Y%m%d_%H%M", gmtime())
            output_dir = './inputs'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            VIDEO_INPUT_FILENAME = ('{}/{}_input.mp4'
                                    .format(output_dir, current_time))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            ret, img = self.cap.read()
            if VIDEO_SOURCE in [0, 1]:
                img = np.rot90(img, 1)
            self.video_writer = cv2.VideoWriter(VIDEO_INPUT_FILENAME,
                                                fourcc, 30,
                                                (img.shape[1], img.shape[0]))
            logger.info("Writing video to {}".format(VIDEO_INPUT_FILENAME))

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
        logger.info("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        if checkpoint is None:
            checkpoint = FLAGS.checkpoint
        logger.info("Checkpoint: {}".format(checkpoint))
        saver.restore(self.sess, checkpoint)

        logger.info("Initialization done")
        """

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

        with tf.Session() as sess:
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

    def run(self):
        while self.run_flag:
            if self.pause_flag:
                time.sleep(0.1)
                continue
            try:
                pose_and_seg_data = (self.pose_and_seg_data_queue
                                     .get(timeout=QUEUE_WAIT_TIME))
            except Empty:
                continue
            start_time = time.time()

            logger.info("VITON inferring ...")
            """
            img = pose_and_seg_data['frame']
            poses = pose_and_seg_data['poses']
            masks = pose_and_seg_data['masks']
            prod_img = np.array(Image.open(self.prod_name))
            output = self.viton_infer(img, prod_img, poses, masks)
            """
            pickle.dump(pose_and_seg_data,
                        open(TMP_DATA_PICKLE_NAME, 'wb'))
            url = VITON_URL
            requests.get(url)
            output = cv2.imread(TMP_OUTPUT_NAME)
            url = ATTACH_URL
            requests.get(url)
            output2 = cv2.imread(TMP_OUTPUT_NAME2)
            self.viton_frame_queue.put(output)
            self.viton_frame_queue2.put(output2)
            logger.info("VITON: {} s"
                        .format(time.time() - start_time))

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class Displayer(threading.Thread):
    """ Thread to display. """

    def __init__(self, frame_queue,
                 pose_frame_queue,
                 segment_frame_queue,
                 viton_frame_queue,
                 viton_frame_queue2,
                 threadManager):
        super(Displayer, self).__init__()
        self.frame_queue = frame_queue
        self.pose_frame_queue = pose_frame_queue
        self.segment_frame_queue = segment_frame_queue
        self.viton_frame_queue = viton_frame_queue
        self.viton_frame_queue2 = viton_frame_queue2
        self.threadManager = threadManager
        self.run_flag = True
        self.pause_flag = False
        if MOTOR_CONTROL:
            arduinoInit()

    def run(self):
        while self.run_flag:
            if self.pause_flag:
                k = cv2.waitKey(25) & 0xFF
                if k == ord('s'):
                    self.threadManager.restart()
                    logger.info("Restarted")
                elif k == ord('q') or k == 27:
                    self.threadManager.stop()
                time.sleep(0.1)
                continue

            # Show input
            try:
                frame = self.frame_queue.get(timeout=1)
                cv2.imshow(ORIGIN_WINDOW_NAME, frame)
                cv2.namedWindow(ORIGIN_WINDOW_NAME)
                cv2.moveWindow(ORIGIN_WINDOW_NAME, 0, 20)
            except Empty:
                logger.warning("Display not getting frames")
                pass
            # Show clothes
            try:
                clothes = cv2.imread('inputs/Trifecta.jpg')
                cv2.imshow(CLOTHES_WINDOW_NAME, clothes)
                cv2.namedWindow(CLOTHES_WINDOW_NAME)
                cv2.moveWindow(CLOTHES_WINDOW_NAME, 0, 700)
            except Exception as err:
                logger.error("Failed to open clothes: {}"
                             .format(err))

            # Show pose
            try:
                posed_img = self.pose_frame_queue.get_nowait()
                cv2.imshow(POSE_WINDOW_NAME, posed_img)
                cv2.namedWindow(POSE_WINDOW_NAME)
                cv2.moveWindow(POSE_WINDOW_NAME, 480, 20)
            except Empty:
                pass
            # Show segment
            try:
                masked_img = self.segment_frame_queue.get_nowait()
                cv2.imshow(SEG_WINDOW_NAME, masked_img)
                cv2.namedWindow(SEG_WINDOW_NAME)
                cv2.moveWindow(SEG_WINDOW_NAME, 960, 20)
            except Empty:
                pass
            # Show VTION
            try:
                viton_img = self.viton_frame_queue.get_nowait()
                cv2.imshow(OUT_WINDOW_NAME, viton_img)
                cv2.namedWindow(OUT_WINDOW_NAME)
                cv2.moveWindow(OUT_WINDOW_NAME, 1440, 20)
            except Empty:
                pass

            # Show Attached
            try:
                attached_img = self.viton_frame_queue2.get_nowait()
                cv2.imshow(OUT2_WINDOW_NAME, attached_img)
                cv2.namedWindow(OUT2_WINDOW_NAME)
                cv2.moveWindow(OUT2_WINDOW_NAME, 1440, 600)
            except Empty:
                pass

            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):
                self.threadManager.pause()
                logger.info("Paused")
                continue
            # Stop the program if 'q' is pressed
            elif k == ord('q') or k == 27:
                self.threadManager.stop()
            elif k == ord('c'):
                self.threadManager.frameReader.put_process()
                logger.info("Put process")
            elif k == ord('a'):
                r = requests.post(CHANGE_CLOTH_URL, data="a.jpg")
                logger.info(r.content)
            elif k == ord('s'):
                r = requests.post(CHANGE_CLOTH_URL, data="s.jpg")
                logger.info(r.content)
            elif k == ord('d'):
                r = requests.post(CHANGE_CLOTH_URL, data="d_0.jpg")
                logger.info(r.content)
            elif k == ord('f'):
                r = requests.post(CHANGE_CLOTH_URL, data="f.jpg")
                logger.info(r.content)
            elif k == ord('1') and MOTOR_CONTROL:
                logger.info("Turn 1")
                digCtrl(1)
            elif k == ord('2') and MOTOR_CONTROL:
                logger.info("Turn 2")
                digCtrl(2)

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class VITONDemo():
    """ Provides main class for real-time VITON demo.

    The class has five threads, FrameReader, SegmentationExtractor,
    PoseEstimator and VITONWorker to go throught the
    whole process. """

    def __init__(self):

        self.batch_size = 1
        self.frame_queue_maxsize = FRAME_QUEUE_MAX_SIZE
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)

        self.process_frame_queue = Queue()
        self.segmentation_data_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.pose_and_seg_data_queue = Queue()

        self.frame_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.pose_frame_queue = Queue()
        self.segment_frame_queue = Queue()
        self.viton_frame_queue = Queue()
        self.viton_frame_queue2 = Queue()

        self.frameReader = FrameReader(self.cap, self.frame_queue,
                                       self.process_frame_queue, self)
        self.segExtractor = SegmentationExtractor(self.process_frame_queue,
                                                  self.segmentation_data_queue,
                                                  self.segment_frame_queue,
                                                  self.batch_size)
        self.poseEstimator = PoseEstimator(self.segmentation_data_queue,
                                           self.pose_and_seg_data_queue,
                                           self.pose_frame_queue)
        self.vitonWorker = VITONWorker(self.pose_and_seg_data_queue,
                                       self.viton_frame_queue,
                                       self.viton_frame_queue2)
        self.displayer = Displayer(self.frame_queue,
                                   self.pose_frame_queue,
                                   self.segment_frame_queue,
                                   self.viton_frame_queue,
                                   self.viton_frame_queue2,
                                   self)

    def run(self):
        logger.info("Threads started")
        self.frameReader.start()
        self.segExtractor.start()
        self.poseEstimator.start()
        self.vitonWorker.start()
        self.displayer.start()

        self.displayer.join()
        self.vitonWorker.join()
        self.poseEstimator.join()
        self.segExtractor.join()
        self.frameReader.join()
        logger.debug("Threads ended")

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("End of video")

    def pause(self):
        self.displayer.pose()
        self.vitonWorker.pause()
        self.poseEstimator.pause()
        self.segExtractor.pause()
        self.frameReader.pause()

    def restart(self):
        self.displayer.restart()
        self.vitonWorker.restart()
        self.poseEstimator.restart()
        self.segExtractor.restart()
        self.frameReader.restart()

    def stop(self):
        self.displayer.stop()
        self.vitonWorker.stop()
        self.poseEstimator.stop()
        self.segExtractor.stop()
        self.frameReader.stop()


if __name__ == "__main__":
    logger.info("Welcome to Liver Failure real-time VITON demo!")
    logger.info("Press q to exit.")
    demo = VITONDemo()
    demo.run()
