from flask import Flask, request
from PIL import Image
import pickle
import numpy as np
import cv2
import logging
from tf_pose_estimation.src.networks import get_graph_path
from tf_pose_estimation.src.estimator import TfPoseEstimator as Pose_Inferrer

app = Flask(__name__)
logger = logging.getLogger(__name__)
(VIDEO_CH, VIDEO_H, VIDEO_W) = (3, 480, 640)


@app.route("/", methods=["POST"])
def home():
    img = Image.open(request.files['files'])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = demo.infer(img)
    img_pickle = pickle.dumps(img)

    return img_pickle


class VITONDemo():
    def __init__(self):
        logger.info("Loading pose_inferrer ...")
        self.pose_inferrer = Pose_Inferrer(get_graph_path('mobilenet_thin'),
                                           target_size=(VIDEO_W, VIDEO_H))
    def infer(self, frame):
        humans = self.pose_inferrer.inference(frame)
        frame = Pose_Inferrer.draw_humans(frame, humans, imgcopy=False)
        return frame

demo = VITONDemo()
