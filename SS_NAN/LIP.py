
from __future__ import division
import os
import time
import numpy as np
import skimage.io
import sklearn.metrics as metrics
from .config import Config
from .utils import Dataset
from  .model import AttResnet101FCN

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/AttResnet101FCN")
FCN = AttResnet101FCN




CLASS_IDS =[1] #change with the config
EXCLUDE_LAYERS=['mask_logits_%d'%i for i in range(4)]
############################################################
#  Configurations
############################################################

class LIPConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "LIP"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU =1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 19 # COCO has 80 classes


############################################################
#  Dataset
############################################################

class LIPDataset(Dataset):
    def load_LIP(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        # Path
        image_dir = os.path.join(dataset_dir,'LIP_dataset', "train_set" if subset == "train"
                                 else "val_set",'images')
        segmentation_dir = os.path.join(dataset_dir,'parsing', "train_segmentations" if subset == "train"
                                 else "val_segmentations")
        cocoseg_dir = os.path.join(dataset_dir,'LIP_dataset', "train_set" if subset == "train"
                                 else "val_set",'cocoseg')
        # Create LIP object
        txt_path_dict = {
            "train": "lists/train_id.txt",
            "val": "lists/val_id.txt",
        }
        image_list = [v.strip() for v in open(os.path.join(dataset_dir,txt_path_dict[subset])).readlines()]
        # Load all classes
        image_ids = list(np.arange(len(image_list)))

        # Add classes
        class_name = ['Hat', 'Hair','Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coats', 'Socks', 'Pants','Jumpsuits',  'Scarf', 'Skirt',  'Face', 'Left-arm','Right-arm',   'Left-leg','Right-leg','Left-shoe','Right-shoe' ]
        for i in range(len(class_name)):
            self.add_class("LIP", i+1, class_name[i])

        # Add images
        for i in image_ids:
            self.add_image(
                "LIP", image_id=i,
                path=os.path.join(image_dir,image_list[i]+'.jpg'),segmentation_img = os.path.join(segmentation_dir,image_list[i]+'.png'))
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "LIP":
            return super(self.__class__).load_mask(image_id)

        instance_masks = []
        class_ids = []
        seg_img = self.image_info[image_id]["segmentation_img"]
        try:
            seg_im = skimage.io.imread(seg_img)
            assert len(seg_im.shape) == 2
            return seg_im
        except:
            print('segmentation cannot be found\n image info:{}'.format(self.image_info[image_id]))
            return super(self.__class__).load_mask(image_id)


############################################################
#  COCO Evaluation
############################################################
def classwise_result(y_true,y_pred,class_nums):
    r=np.zeros([class_nums,3])
    for v in range(class_nums):
        r[v][0]=np.sum((y_true==v)&(y_pred==v))
        r[v][1]=np.sum((y_true==v)&(y_pred!=v))
        r[v][2]=np.sum((y_true!=v)&(y_pred==v))
    return r
def evaluate_LIP(dataset,limit=0):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load imagei
        image = dataset.load_image(image_id)
        gt_mask = dataset.load_mask(image_id)
        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        results.append(classwise_result(gt_mask.flatten(),r['probs'].argmax(-1).flatten(),len(dataset.class_ids)))
        #results.append( classwise_result(gt_mask.flatten(),r['masks'].flatten(),len(dataset.class_ids)))
    # Load results. This modifies results with additional attributes.

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    return np.array(results)

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train FCN on LIP.')
    parser.add_argument("command",
                        metavar="<command>",default='train',
                        help="'train' or 'evaluate' on LIP")
    parser.add_argument('--dataset',default ='/media/ltp/40BC89ECBC89DD32/LIPHP_data/LIP/SinglePerson',
                        metavar="/path/to/coco/",
                        help='Directory of the LIPdataset')
    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",default='./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--exclude',
                        default=False,
                        help="exclude classify layers")
    parser.add_argument('--trainmode',
                            default='finetune',
                            help="train sub mode")
    parser.add_argument('--evalnum', type=int,
                                default=0,
                                help="eval image num")
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "train":
        config = LIPConfig()
    else:
        class InferenceConfig(LIPConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            IMAGE_MIN_DIM =120
            IMAGE_MAX_DIM =640
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = FCN(mode="training", config=config,
                                  model_dir=MODEL_DIR,trainmode=args.trainmode)
    else:
        model = FCN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "pspnet":
        model_path = './att_pspnet.h5'
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path ='./pspnet101_voc2012.h5'
        if config.GPU_COUNT>1:
            v=[v for v in model.keras_model.inner_model.layers if v.name=='resnet101_model']
        else:
            v=[v for v in model.keras_model.layers if v.name=='resnet101_model']
        v[0].load_weights(model_path,by_name=True)
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.exclude:
        model.load_weights(model_path, by_name=True,exclude=EXCLUDE_LAYERS)
    else:
        model.load_weights(model_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = LIPDataset()
        dataset_train.load_LIP(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = LIPDataset()
        dataset_val.load_LIP(args.dataset, "val")
        dataset_val.prepare()

        # This training schedule is an example. Update to fit your needs.

        # Training - Stage 1
        # Adjust epochs and layers as needed
        config.STEPS_PER_EPOCH =round(len(dataset_train.image_info)/config.BATCH_SIZE)
        config.VALIDATION_STPES=round(config.STEPS_PER_EPOCH/10)
        print("Training network 5+")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,layers='head')
        config.BATCH_SIZE=10*config.GPU_COUNT
        config.STEPS_PER_EPOCH =round(len(dataset_train.image_info)/config.BATCH_SIZE)
        config.VALIDATION_STPES=round(config.STEPS_PER_EPOCH/10)
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,layers='psp5+')
#'''Need to rewrite'''

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = LIPDataset()
        dataset_val.load_LIP(args.dataset, "val")
        dataset_val.prepare()
        results = evaluate_LIP(dataset_val,args.evalnum)
        # TODO: evaluating
        overall_samples_results = np.sum(results,0)
        Acc = np.sum(overall_samples_results[:,0])/np.sum(overall_samples_results[:,:2])
        class_acc=overall_samples_results[:,0]/(overall_samples_results[:,0]+overall_samples_results[:,1])
        meanAcc = np.mean(class_acc,0)
        IOU =overall_samples_results[:,0]/np.sum(overall_samples_results,1)
        meanIou = np.mean(IOU,0)
        print('Accuracy: {x:.4f}  meanAcc: {y:.4f} meanIou: {z:.4f}'.format(x=Acc,y=meanAcc,z=meanIou))
        for v in range(class_acc.shape[0]):
            print('Classname:{x:}   Acc: {y:.4f}  Iou: {z:.4f}'.format(x=dataset_val.class_names[v],y=class_acc[v],z=IOU[v]))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
