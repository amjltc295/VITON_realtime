""" Provides class for real-time VITON demo. """
import logging
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import threading
import time
from time import gmtime, strftime
from queue import Queue, Empty
import yaml
import os
import socket
import pickle

import SS_NAN.LIP
import SS_NAN.visualize as visualize
from SS_NAN.model import AttResnet101FCN as Seg_Inferrer
from tf_pose_estimation.src.networks import get_graph_path
from tf_pose_estimation.src.estimator import TfPoseEstimator as Pose_Inferrer
from model_zalando_mask_content import create_model as VITON_Inferrer

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

# Set up user parameters
with open("./configure.yml", 'r') as stream:
    user_parameters = yaml.load(stream)

###################
# User Parameters #
###################
VIDEO_SOURCE = user_parameters['video_source']
VIDEO_FPS = user_parameters['video_fps']
SKIP_FRAME = user_parameters['skip_frame']
SKIP_FRAME_RATIO = user_parameters['skip_frame_ratio']
DRAW_BBOX = user_parameters['draw_bbox']
TRANSPARENCY_ALPHA = user_parameters['transparency_alpha']
WINDOW_NAME = user_parameters['window_name']
FULL_SCREEN = user_parameters['full_screen']
DISPLAY_MODE = user_parameters['display_mode']
SHOW_ID = user_parameters['show_id']

WRITE_VIDEO_OUTPUT = user_parameters['write_video_output']
WRITE_VIDEO_INPUT = user_parameters['write_video_input']
if WRITE_VIDEO_OUTPUT:
    # Create directories for output if not exists
    output_dir = './outputs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if VIDEO_SOURCE == 0 or VIDEO_SOURCE == 1:
        current_time = strftime("%Y%m%d_%H%M", gmtime())
        VIDEO_OUTPUT_FILENAME = ('{}/{}_iseg_demo.mp4'
                                 .format(output_dir, current_time))
    else:
        VIDEO_OUTPUT_FILENAME = (output_dir
                                 + VIDEO_SOURCE.split('/')[1].split('.')[0]
                                 + '_out.mp4')
if WRITE_VIDEO_OUTPUT:
    # Create directories for output if not exists
    output_dir = './outputs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if VIDEO_SOURCE == 0 or VIDEO_SOURCE == 1:
        current_time = strftime("%Y%m%d_%H%M", gmtime())
        VIDEO_INPUT_FILENAME = ('{}/{}_iseg_demo_input.mp4'
                                .format(output_dir, current_time))
    else:
        WRITE_VIDEO_INPUT = False

##################################
# Video and Queue I/O Parameters #
##################################

VIDEO_SIZE = (VIDEO_CH, VIDEO_H, VIDEO_W) = (3, 480, 640)
FRAME_QUEUE_MAX_SIZE = 2
ISEG_BATCH_SIZE = 1
QUEUE_WAIT_TIME = 4

############################
# Visualization Parameters #
############################

# All in configuration.yml

#####################
# SS_NAN Parameters #
#####################
SS_NAN_MODEL_DIR = './SS_NAN/model/logs'
LIP_MODEL_PATH = './SS_NAN/AttResnet101FCN_lip_0023.h5'


class SegInferenceConfig(SS_NAN.LIP.LIPConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MAX_DIM = 640


seg_config = SegInferenceConfig()
seg_config.display()

#################
# Socket Server #
#################
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
logger.debug(host)
serversocket.bind(('140.112.29.182', 8089))
serversocket.listen(2)  # become a server socket, maximum 5 connections


while True:
    logger.info("Waiting for connection ..")
    connection, address = serversocket.accept()
    logger.info("{} connected".format(address))
    break

msg = connection.recv(100000)
logger.info(msg)


class FrameReader(threading.Thread):
    """ Thread to read frame from webcam.
    Need to be stopped by calling stop() in IdentifyAndDisplayer."""

    def __init__(self, cap, frame_queue, threadManager):
        super(FrameReader, self).__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.threadManager = threadManager
        self.run_flag = True
        self.pause_flag = False

    def run(self):
        count = 0
        while self.run_flag:
            if self.pause_flag:
                time.sleep(0.1)
                continue
            start_time = time.time()
            count += 1
            buff = []
            while True:
                packet = connection.recv(921780)
                if packet == b'': break
                if not packet: break
                print(len(packet))
                buff.append(packet)
            frame = pickle.loads(b"".join(buff))
            """
            buf = connection.recv(921780)
            print(buf)
            print(len(buf))
            frame = pickle.loads(buf)
            """
            frame = frame['frame']
            if frame is None:
                logger.warning("Not getiing frames")
                self.stop()
                continue
            if SKIP_FRAME and count % SKIP_FRAME_RATIO != 0:
                continue
            if self.frame_queue.full():
                self.frame_queue.get()
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            self.frame_queue.put(frame)
            logger.debug("Frame read: {} s".format(time.time() - start_time))

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class SegmentationExtractor(threading.Thread):
    """ Thread to do instance segmentaion. """

    def __init__(self, frame_queue, segmentation_data_queue,
                 seg_inferrer, batch_size):
        super(SegmentationExtractor, self).__init__()
        self.frame_queue = frame_queue
        self.segmentation_data_queue = segmentation_data_queue
        self.seg_inferrer = seg_inferrer
        self.run_flag = True
        self.pause_flag = False
        self.batch_size = batch_size
        self.class_names =  ['BG', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                             'Upper-clothes', 'Dress', 'Coats', 'Socks',
                             'Pants','Jumpsuits',  'Scarf', 'Skirt',
                             'Face', 'Left-arm','Right-arm', 'Left-leg',
                             'Right-leg', 'Left-shoe', 'Right-shoe']


    def run(self):
        while self.run_flag:
            if self.pause_flag:
                time.sleep(0.1)
                continue
            start_time = time.time()
            frames = []
            for i in range(self.batch_size):
                try:
                    frame = self.frame_queue.get(timeout=QUEUE_WAIT_TIME)
                except Empty:
                    logger.warning("segExtractor not getting frames")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            if frames == []:
                continue

            logger.debug("Seg-pre: {} s".format(time.time() - start_time))
            start_time = time.time()
            try:
                results = self.seg_inferrer.detect(frames, verbose=0)
                masks = results[0]['masks']
                color = visualize.random_colors(N=seg_config.NUM_CLASSES)
                frame = visualize.apply_mask(frame, masks, color=color,
                                             class_ids=[v for v in
                                                        range(1, seg_config.NUM_CLASSES)])

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
                 pose_inferrer):
        super(PoseEstimator, self).__init__()
        self.segmentation_data_queue = segmentation_data_queue
        self.pose_and_seg_data_queue = pose_and_seg_data_queue
        self.pose_inferrer = pose_inferrer
        self.run_flag = True
        self.pause_flag = False

    def run(self):
        while self.run_flag:
            if self.pause_flag:
                time.sleep(0.1)
                continue
            try:
                segmentation_data = (self.segmentation_data_queue
                                     .get(timeout=QUEUE_WAIT_TIME))
            except Empty:
                logger.warning("PoseEstimator not getting frames")
                continue
            start_time = time.time()
            if 'masks' not in segmentation_data:
                self.pose_and_seg_data_queue.put(segmentation_data)
                logger.debug("Pose infer: {} s".format(time.time() - start_time))
                continue

            frame = segmentation_data['frame']
            humans = self.pose_inferrer.inference(frame)

            segmentation_data['humans'] = humans
            frame = Pose_Inferrer.draw_humans(frame, humans, imgcopy=False)
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
                 viton_data_queue, viton_worker):
        super(VITONWorker, self).__init__()
        self.pose_and_seg_data_queue = pose_and_seg_data_queue
        self.viton_data_queue = viton_data_queue
        self.viton_worker = viton_worker
        self.run_flag = True
        self.pause_flag = False

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

            # If there are no people in the frame, just display it.
            if 'cropped_images' not in pose_and_seg_data:
                self.viton_data_queue.put(pose_and_seg_data)
                logger.debug("VITON: {} s"
                             .format(time.time() - start_time))
                continue

            # Re-identification process
            feature_wrappers = self.bind_ID_to_image_info(pose_and_seg_data)
            self.update_tracker(feature_wrappers)
            # self.match_ID(feature_wrappers,
            #               self.last_re_id_feature_wrappers)

            # Update stored ID feature
            self.last_re_id_feature_wrappers = feature_wrappers

            pose_and_seg_data['re_IDed_feature_wrappers'] = \
                feature_wrappers
            pose_and_seg_data['tracks'] = self.tracker.tracks
            self.viton_data_queue.put(pose_and_seg_data)
            viton_data_pickle = pickle.dumps(pose_and_seg_data)
            connection.send(viton_data_pickle)

            logger.debug("VITON: {} s"
                         .format(time.time() - start_time))

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class VITONDemo():
    """ Provides main class for real-time VITON demo.

    The class has five threads, FrameReader, SegmentationExtractor,
    PoseEstimator, VITONWorker and Displayer to go throught the
    whole process. """

    def __init__(self, video_source):
        logger.info("Loading pose_inferrer ...")
        self.pose_inferrer = Pose_Inferrer(get_graph_path('mobilenet_thin'),
                                           target_size=(VIDEO_W, VIDEO_H))

        """
        logger.info("Creating seg_inferrer ...")
        self.seg_inferrer = Seg_Inferrer(mode="inference",
                                         model_dir=SS_NAN_MODEL_DIR,
                                         config=seg_config)
        logger.info("Loading seg_inferrer weight...")
        self.seg_inferrer.load_weights(LIP_MODEL_PATH, by_name=True)
        self.seg_inferrer.keras_model._make_predict_function()

        logger.info("Loading viton_inferrer ...")
        self.viton_inferrer = VITON_Inferrer()
        """
        self.seg_inferrer = None
        self.viton_inferrer = None

        self.batch_size = 1
        self.frame_queue_maxsize = FRAME_QUEUE_MAX_SIZE
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)

        self.frame_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.segmentation_data_queue = Queue(maxsize=self.frame_queue_maxsize)
        self.pose_and_seg_data_queue = Queue()
        self.viton_data_queue = Queue()

        self.frameReader = FrameReader(self.cap, self.frame_queue, self)
        self.segExtractor = SegmentationExtractor(self.frame_queue,
                                                  self.segmentation_data_queue,
                                                  self.seg_inferrer,
                                                  self.batch_size)
        self.poseEstimator = PoseEstimator(self.segmentation_data_queue,
                                           self.pose_and_seg_data_queue,
                                           self.pose_inferrer)
        self.vitonWorker = VITONWorker(self.pose_and_seg_data_queue,
                                       self.viton_data_queue,
                                       self.viton_inferrer)
        connection.send(b'start')

    def run(self):
        logger.debug("Threads started")
        self.frameReader.start()
        self.segExtractor.start()
        self.poseEstimator.start()
        self.vitonWorker.start()

        self.vitonWorker.join()
        self.poseEstimator.join()
        self.segExtractor.join()
        self.frameReader.join()
        logger.debug("Threads ended")

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("End of video")

    def pause(self):
        self.vitonWorker.pause()
        self.poseEstimator.pause()
        self.segExtractor.pause()
        self.frameReader.pause()

    def restart(self):
        self.vitonWorker.restart()
        self.poseEstimator.restart()
        self.segExtractor.restart()
        self.frameReader.restart()

    def stop(self):
        self.vitonWorker.stop()
        self.poseEstimator.stop()
        self.segExtractor.stop()
        self.frameReader.stop()


if __name__ == "__main__":
    logger.info("Welcome to Liver Failure real-time VITON demo!")
    logger.info("Press q to exit.")
    demo = VITONDemo(video_source=VIDEO_SOURCE)
    demo.run()
