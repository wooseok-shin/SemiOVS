# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import cv2
import torch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from sed import add_sed_config

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


# background (truck, traffic light, ... -> background) & foreground class
coco2voc_classes = [
               ["truck", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "bed", "toilet", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"],
               ['aeroplane'], ['bicycle'], ['bird'], ['boat'], ['bottle'],['bus'], ['car'], 
               ['cat'], ['chair'], ['cow'], ['dining table'], ['dog'],['horse'], ['motorbike'], 
               ['person'], ['potted plant'], ['sheep'], ['sofa'], ['train'], ['tv/monitor']
               ]

coco2pc60_classes = [
    ["fire hydrant", "parking meter", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "toilet", "laptop", "remote", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "bush", "cage", "cardboard", "carpet", "clouds", "counter", "cupboard", "desk-stuff", "dirt", "fog", "fruit", "furniture-other", "gravel", "hill", "house", "leaves", "mat", "metal", "mirror-stuff", "moss", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "playingfield", "railing", "railroad", "river", "roof", "rug", "salad", "sand", "sea", "shelf", "solid-other", "stairs", "stone", "straw", "structural-other", "coffee table", "side table", "tent", "textile-other", "towel", "vegetable"],
    ["aeroplane"], ["bag"], ["bed"], ["bedclothes"], ["bench"], ["bicycle"], ["bird"], ["boat"], ["book"], ["bottle"], ["building"], ["bus"], 
    ["cabinet"], ["car"], ["cat"], ["ceiling"], ["chair"], ["cloth"], ["computer"], ["cow"], ["cup"], ["curtain"], ["dog"], ["door"], ["fence"],
    ["floor"], ["flower"], ["food"], ["grass"], ["ground"], ["horse"], ["keyboard"], ["light"], ["motorbike"], ["mountain"], ["mouse"], ["person"],
    ["plate"], ["platform"], ["potted plant"], ["road"], ["rock"], ["sheep"], ["shelves"], ["sidewalk"], ["sign"], ["sky"], ["snow"], ["sofa"],
    ["dining table"], ["track"], ["train"], ["tree"], ["truck"], ["tv/monitor"], ["wall"], ["water"], ["window"], ["wood"]
]

VOC_PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], *[[255, 255, 255] for i in range(256 - 21 - 1)],
    [224, 224, 192]], dtype=np.uint8)


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def aggregate_concept_predictions(pred, class_to_concept_idxs):
    _, H, W = pred.shape
    agg_pred = torch.zeros(len(class_to_concept_idxs), H, W, device=pred.device)
    for cls_i, conc_i in class_to_concept_idxs.items():
        agg_pred[cls_i] = pred[conc_i].max(dim=0).values
    return agg_pred

def flatten_class_concepts(class_concepts):
    concepts = []
    concept_to_class_idx = {}
    class_to_concept_idxs = {}
    for i, cls_concepts in enumerate(class_concepts):
        for concept in cls_concepts:
            concept_to_class_idx[len(concepts)] = i
            if i not in class_to_concept_idxs:
                class_to_concept_idxs[i] = []
            class_to_concept_idxs[i].append(len(concepts))
            concepts.append(concept)
    return concepts, concept_to_class_idx, class_to_concept_idxs


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")
    predictor = DefaultPredictor(cfg)

    dataset_type = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON

    if 'voc' in dataset_type:
        with open('unlabeled_coco_for_pascal.txt', 'r') as f:
            ids = f.read().splitlines()
    elif 'pc60' in dataset_type:
        with open('unlabeled_coco_for_pc60.txt', 'r') as f:
            ids = f.read().splitlines()
    ids.sort()
    print('number of OOD images:', len(ids))

    root = '../data/coco/'
    for i, id in enumerate(tqdm(ids)):
        img = read_image(os.path.join(root, id.split(' ')[0]), format='BGR')
        predictions = predictor(img)["sem_seg"]

        ## Aggregate
        if 'voc' in dataset_type:
            _, _, cls2con = flatten_class_concepts(coco2voc_classes)   # coco2voc_classes
            palette = VOC_PALETTE
        elif 'pc60' in dataset_type:
            _, _, cls2con = flatten_class_concepts(coco2pc60_classes)   # coco2pc60_classes
            palette = get_palette(60)

        predictions = aggregate_concept_predictions(predictions, cls2con)
        predictions_conf, predictions = predictions.max(dim=0)
        assert img.shape[:2] == predictions.shape, (img.shape, predictions.shape)
        
        pl_mask = predictions.detach().cpu().numpy().astype(np.uint8)

        pl_mask = Image.fromarray(pl_mask).convert('P')
        pl_mask.putpalette(palette)
        os.makedirs(os.path.join(root, os.path.dirname(id.split(' ')[1])), exist_ok=True)
        pl_mask.save(os.path.join(root, id.split(' ')[1]))
        

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
