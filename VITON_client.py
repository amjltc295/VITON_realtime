""" Provides class for real-time VITON demo. """
import logging
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import threading
import time
from time import gmtime, strftime
import yaml
import os
import socket
import pickle

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

#################
# Socket Clinet #
#################
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
logger.info("Connecting to server ...")
clientsocket.connect(('140.112.29.182', 8089))


class FrameReader(threading.Thread):
    """ Thread to read frame from webcam.
    Need to be stopped by calling stop() in IdentifyAndDisplayer."""

    def __init__(self, cap, threadManager):
        super(FrameReader, self).__init__()
        self.cap = cap
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
            ret, frame = self.cap.read()
            if frame is None:
                logger.warning("Not getiing frames")
                self.stop()
                continue
            if SKIP_FRAME and count % SKIP_FRAME_RATIO != 0:
                continue
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            frame_dict = {'frame': frame}
            frame_pickle = pickle.dumps(frame_dict)
            print(len(frame_pickle))

            clientsocket.send(frame_pickle)
            logger.debug("Frame read: {} s".format(time.time() - start_time))

    def stop(self):
        self.run_flag = False

    def pause(self):
        self.pause_flag = True

    def restart(self):
        self.pause_flag = False


class Displayer(threading.Thread):
    """ Thread to draw the results and display on the screen. """

    def __init__(self, threadManager):
        super(Displayer, self).__init__()
        self.threadManager = threadManager
        self.run_flag = True
        self.pause_flag = False
        self.font_size = 20
        self.fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf',
                                      self.font_size)
        if WRITE_VIDEO_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            self.video_out = cv2.VideoWriter(VIDEO_OUTPUT_FILENAME,
                                             fourcc, VIDEO_FPS,
                                             (VIDEO_W, VIDEO_H))

        if WRITE_VIDEO_INPUT:
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            self.video_in = cv2.VideoWriter(VIDEO_INPUT_FILENAME,
                                            fourcc, VIDEO_FPS,
                                            (VIDEO_W, VIDEO_H))

    def draw_box_mask_ID(self, frame, tracks, feature_wrappers=None):
        """ Draw bounding box, instance segmentation mask and ID for each
        detected person. """
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        for i in tracks:
            mask = i.iseg_mask
            box = i.bbox
            """
            box = i.to_tlbr()
            i_detected = False
            # Fix Kalman Filter bounding box shift
            if feature_wrappers is not None:
                for j in feature_wrappers:
                    if j.ID == i.track_id:
                        print("{}, {} / {}".format(j.ID, box, j.bbox))
                        mask = j.mask
                        box = j.bbox
                        i_detected = True
                        break

            if not i_detected:
                print("{}, {} ".format(i.track_id, box))
            """
            h, w = mask.shape
            text_on_screen = '{}'.format(i.track_id)
            my_color = (255, 255, 0)

            box = [float(c) for c in box]
            if DRAW_BBOX:
                draw.rectangle([int(box[0]), int(box[1]),
                                int(box[2]), int(box[3])],
                               outline=my_color)

            if SHOW_ID:
                draw.text((int(box[0]+box[2])/2, int(box[1]) - self.font_size),
                          text_on_screen,
                          font=self.fnt, fill=my_color)

        del draw
        frame = np.array(frame)
        return frame

    def run(self):
        if FULL_SCREEN:
            cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(WINDOW_NAME)
            cv2.resizeWindow(WINDOW_NAME, VIDEO_W*2, VIDEO_H*2)

        while self.run_flag:
            if self.pause_flag:
                k = cv2.waitKey(1)
                if k == ord('s'):
                    self.threadManager.restart()
                    logger.info("Restarted")
                elif k == ord('q') or k == 27:
                    self.threadManager.stop()
                time.sleep(0.1)
                continue
            start_time_fps = time.time()
            k = cv2.waitKey(1)
            if k == ord('p'):
                self.threadManager.pause()
                logger.info("Paused")
                continue
            # Stop the program if 'q' is pressed
            elif k == ord('q') or k == 27:
                self.threadManager.stop()

            try:
                viton_data_pickle = clientsocket.recv(10000000)
                viton_data = pickle.loads(viton_data_pickle)

            except Exception as err:
                logger.error(err)
                continue
            start_time = time.time()
            frame = viton_data['frame']
            if 'cropped_images' not in viton_data:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                fps = "FPS: {:.1f}".format(1.0 / (time.time()-start_time_fps))
                cv2.putText(frame, fps, (30, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                cv2.imshow(WINDOW_NAME, frame)
                continue
            feature_wrappers = \
                viton_data['re_IDed_feature_wrappers']

            tracks = viton_data['tracks']

            if WRITE_VIDEO_INPUT:
                self.video_in.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # Draw result
            # frame = self.draw_box_mask_ID(frame, feature_wrappers)
            frame = self.draw_box_mask_ID(frame, tracks, feature_wrappers)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = "FPS: {:.1f}".format(1.0 / (time.time() - start_time_fps))
            cv2.putText(frame, fps, (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
            if WRITE_VIDEO_OUTPUT:
                self.video_out.write(frame)
            cv2.imshow(WINDOW_NAME, frame)

            logger.debug("Display: {} s".format(time.time() - start_time))
        if WRITE_VIDEO_INPUT:
            logger.info("Input: {} written.".format(VIDEO_INPUT_FILENAME))
            self.video_in.release()
        if WRITE_VIDEO_OUTPUT:
            logger.info("Output: {} written.".format(VIDEO_OUTPUT_FILENAME))
            self.video_out.release()

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
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)

        self.frameReader = FrameReader(self.cap, self)
        self.displayer = Displayer(self)
        clientsocket.send(b'hi from client')
        while True:
            msg = clientsocket.recv(10000)
            logger.info(msg)
            if msg == b'start':
                break

    def run(self):
        logger.debug("Threads started")
        self.frameReader.start()
        self.displayer.start()

        self.displayer.join()
        self.frameReader.join()
        logger.debug("Threads ended")

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("End of video")

    def pause(self):
        self.displayer.pause()
        self.frameReader.pause()

    def restart(self):
        self.displayer.restart()
        self.frameReader.restart()

    def stop(self):
        self.displayer.stop()
        self.frameReader.stop()


if __name__ == "__main__":
    logger.info("Welcome to Liver Failure real-time VITON demo!")
    logger.info("Press q to exit.")
    demo = VITONDemo(video_source=VIDEO_SOURCE)
    demo.run()
