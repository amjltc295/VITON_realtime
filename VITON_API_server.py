from flask import Flask, request
from PIL import Image
import pickle
import numpy as np
import cv2
import logging
from tf_pose_estimation.src.networks import get_graph_path
from tf_pose_estimation.src.estimator import TfPoseEstimator as Pose_Inferrer
import SS_NAN.LIP
import SS_NAN.visualize as visualize
from SS_NAN.model import AttResnet101FCN as Seg_Inferrer

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


class SegInferenceConfig(SS_NAN.LIP.LIPConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MAX_DIM = 640


seg_config = SegInferenceConfig()
seg_config.display()


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

    def pose_infer(self, frame):
        humans = self.pose_inferrer.inference(frame)
        frame = Pose_Inferrer.draw_humans(frame, humans, imgcopy=False)
        return frame

    def seg_infer(self, frame):
        results = self.seg_inferrer.detect([frame], verbose=0)
        masks = results[0]['masks']
        color = visualize.random_colors(N=seg_config.NUM_CLASSES)
        frame = visualize.apply_mask(frame, masks, color=color,
                                     class_ids=[v for v in
                                                range(1, seg_config.NUM_CLASSES)])
        return frame


demo = VITONDemo()
app = Flask(__name__)


@app.route("/", methods=["POST"])
def home():
    img = Image.open(request.files['files'])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = demo.pose_infer(img)
    img = demo.seg_infer(img)
    img_pickle = pickle.dumps(img)

    return img_pickle
