"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 dsb2018.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 dsb2018.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 dsb2018.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 dsb2018.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 dsb2018.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np

from config import Config
import utils
import model as modellib

import skimage
import pandas as pd

# Disallow eager use of GPU memory
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
tf.Session(config=config)

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_dsb2018")
# DEFAULT_DATASET_YEAR = "2014"

# Datasets for Kaggle Data Science Bowl 2018
TRAIN_PATH = '/home/yschoi/work/challenges/Kaggle_Data_Science_Bowl_2018/input/stage1_train/'
TEST_PATH = '/home/yschoi/work/challenges/Kaggle_Data_Science_Bowl_2018/input/stage1_test/'

############################################################
#  Configurations
############################################################

class Dsb2018Config(Config):
    NAME = "dsb2018"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512 # 1024?
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 500
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT)
    MEAN_PIXEL = [0, 0, 0]
    LEARNING_RATE = 0.01
    USE_MINI_MASK = True
    MAX_GT_INSTANCES = 500
    # # Minimum probability value to accept a detected instance
    # # ROIs below this threshold are skipped
    # DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class Dsb2018Dataset(utils.Dataset):

    def load_dsb2018(self, dataset_dir, data_type):
        # Add classes
        self.add_class("dsb2018", 1, "nucleus")

        # Add images
        data_ids = next(os.walk(dataset_dir))[1]
        data_split = np.int32(len(data_ids)*0.9)      # 10% validataion
        data_start = 0

        if data_type=="train":
            data_ids = data_ids[:data_split]
        elif data_type=="val":
            data_ids = data_ids[data_split:]
            data_start = data_split
        else:   # "test" and othres
            pass

        for i, id_ in enumerate(data_ids, start=data_start):
            img_path = dataset_dir + id_ + '/images/' + id_ + '.png'
            self.add_image(
                source="dsb2018", image_id=i, path=img_path,
                mask_dir=dataset_dir + id_ + '/masks/',
                width=256, height=256     # Dummy placeholders (will be updated in 'load_image()')
            )

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # Add
        self.image_info[image_id]["height"] = image.shape[0]
        self.image_info[image_id]["width"]  = image.shape[1]

        # If grayscale. Convert to RGB for consistency.
        if image.ndim < 3:
            image = skimage.color.gray2rgb(image)
        return image[:,:,:3]

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []

        for mask_file in next(os.walk(image_info['mask_dir']))[2]:
            mask_ = skimage.io.imread(image_info['mask_dir'] + mask_file)
            # mask_ = mask_[:, :, np.newaxis]
            instance_masks.append(mask_)
            class_ids.append(1)     # Only one class with 'class_id'=1

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(Dsb2018Dataset, self).load_mask(image_id)


############################################################
#  DSB-2018 Evaluation
############################################################

def build_dsb2018_results(dataset, image_ids, rois, class_ids, scores, masks):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "dsb2018"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

def evaluate_dsb2018(model, dataset, eval_type="bbox", limit=0, image_ids=None):
    """Runs evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding image IDs.
    dsb2018_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        pred_masks_combined = np.max(r['masks'], axis=2)
        # skimage.io.imshow(pred_masks_combined)

        results.append(pred_masks_combined)

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

    return results


# Define a run-length encoding function
from skimage.morphology import label

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2

    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return run_lengths

def generate_rle_code(x):
    lab_img = label(x)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


############################################################
#  Training
############################################################

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on DSB-2018 dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on DSB-2018")
    # parser.add_argument('--dataset', required=True,
    #                     metavar="/path/to/coco/",
    #                     help='Directory of the MS-COCO dataset')
    # parser.add_argument('--year', required=False,
    #                    default=DEFAULT_DATASET_YEAR,
    #                    metavar="<year>",
    #                    help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=65,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    # parser.add_argument('--download', required=False,
    #                    default=False,
    #                    metavar="<True|False>",
    #                    help='Automatically download and unzip MS-COCO files (default=False)',
    #                    type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    # print("Dataset: ", args.dataset)
    #print("Year: ", args.year)
    print("Logs: ", args.logs)
    #print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        # config = CocoConfig()
        config = Dsb2018Config()
    else:
        #class InferenceConfig(CocoConfig):
        class InferenceConfig(Dsb2018Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    excluded_layers = []
    if args.model.lower() == "coco":    # Transfer learning from "coco"
        model_path = COCO_MODEL_PATH
        excluded_layers=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_mask']
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    # model.load_weights(model_path, by_name=True)
    model.load_weights(model_path, by_name=True, exclude=excluded_layers)

    # Train or evaluate
    if args.command == "train":
        # Training dataset.  
        dataset_train = Dsb2018Dataset()
        dataset_train.load_dsb2018(TRAIN_PATH, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = Dsb2018Dataset()
        dataset_val.load_dsb2018(TRAIN_PATH, "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_test = Dsb2018Dataset()
        dataset_test.load_dsb2018(TEST_PATH, "test")
        dataset_test.prepare()
        print("Running DSB-2018 evaluation on {} images.".format(args.limit))
        evaluate_dsb2018(model, dataset_test, "bbox", limit=int(args.limit))

    elif args.command == "submit":
        # Validation dataset
        dataset_test = Dsb2018Dataset()
        dataset_test.load_dsb2018(TEST_PATH, "test")
        dataset_test.prepare()
        print("Running DSB-2018 evaluation on {} images.".format(args.limit))
        pred_results = evaluate_dsb2018(model, dataset_test, "bbox")

        test_ids = next(os.walk(TEST_PATH))[1]
        new_test_ids = []
        rles = []
        for n, id_ in enumerate(test_ids):
            rle = list(generate_rle_code(pred_results[n]))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))

        import datetime

        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('./output/mask_rcnn_submission_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv', index=False)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
