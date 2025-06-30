
## Prepare datasets

Please download the datasets (using wget) and extract them to `./data/`.

- Pascal VOC: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://github.com/wooseok-shin/SemiOVS/releases/download/preliminary/ground_truth_pascal_voc.zip)
- Pascal Context: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) | [SegmentationClassContext60](https://github.com/wooseok-shin/SemiOVS/releases/download/preliminary/ground_truth_pascal_context.zip)

- COCO (Out-of-Distribution data): [train2017](http://images.cocodataset.org/zips/train2017.zip)


The final folder structure should look like this:
```
├── data
	├── pascal
		├── JPEGImages
		└── SegmentationClass
	├── pascal_context
		├── JPEGImages
		└── SegmentationClassContext60
            ├── train
            └── val
	├── coco
		├── images/train2017
		├── anntations_pseudo (from OVS)
            ├── pseudo_labels_for_pascal
            └── pseudo_labels_for_pc60
```

