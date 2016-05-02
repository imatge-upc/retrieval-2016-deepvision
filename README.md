# Faster R-CNN Features for Instance Search

This python repository contains the necessary tools to reproduce the retrieval pipeline based on off-the-shelf Faster R-CNN features described in:

Amaia Salvador, Xavier Giró-i-Nieto, Ferran Marqués, Shin'ichi Satoh, Faster R-CNN Features for Instance Search
Accepted at the DeepVision CVPR Workshop 2016. You can find a preprint of this work [here](http://arxiv.org/abs/1604.08893).

### Setup

- You need to download and install Faster R-CNN [python implementation by Ross Girshick](https://github.com/rbgirshick/py-faster-rcnn). Point ```params['fast_rcnn_path']``` to the Faster R-CNN root path.
- Download [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) Buildings datasets. There are scripts under ```data/images/paris``` and ```data/images/oxford/``` that will do that for you.
- Download Faster R-CNN models by running ```data/models/fetch_models.sh```.

### Usage

- Data preparation. Run ```read_data.py``` to create the lists of query and database images. Run this twice changing ```params['dataset']``` to ```'oxford'``` and ```'paris'```.
- Feature Extraction. Run ```features.py``` to extract Fast R-CNN features for all images in a dataset and store them to disk.
- Ranking. Run ```ranker.py``` to generate and store the rankings for the queries of the chosen dataset.
- Rerank based on region features by running ```rerank.py```.
- Evaluation. Run ```eval.py``` to obtain the Average Precision.
- Visualization. Run ```vis.py```to populate ```data/figures``` with the visualization of the top generated rankings for each query.
