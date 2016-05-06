# Faster R-CNN Features for Instance Search

|  ![CVPR 2016 logo][logo-cvpr] | Paper accepted at [2016 IEEE Conference on Computer Vision and Pattern Recognition Workshops](http://www.deep-vision.net/)   |
|:-:|---|

[logo-cvpr]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/logos/deepvision.png "DeepVision CVPRW 2016 logo"

| ![Amaia Salvador][salvador-photo]  | ![Xavier Giro-i-Nieto][giro-photo]  | ![Ferran Marqués][marques-photo]  | ![Shin'ichi Satoh][satoh-photo]  |
|:-:|:-:|:-:|:-:|:-:|
| [Amaia Salvador][salvador-web]  | [Xavier Giro-i-Nieto][giro-web]  |  [Ferran Marques][marques-web] | [Shin'ichi Satoh][satoh-web]  |


[salvador-web]: https://imatge.upc.edu/web/people/amaia-salvador
[giro-web]: https://imatge.upc.edu/web/people/xavier-giro
[satoh-web]: http://research.nii.ac.jp/~satoh/
[marques-web]:https://imatge.upc.edu/web/people/ferran-marques

[salvador-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/authors/salvador.jpg "Amaia Salvador"
[giro-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/authors/giro.jpg "Xavier Giro-i-Nieto"
[marques-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/authors/marques.jpg "Ferran Marques"
[satoh-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/authors/satoh.jpg "Shin'ichi Satoh"

A joint collaboration between:

|![logo-upc] | ![logo-etsetb] | ![logo-gpi] |  ![logo-nii] |
|:-:|:-:|:-:|:-:|
|[Universitat Politecnica de Catalunya (UPC)][upc-web]   | [UPC ETSETB TelecomBCN][etsetb-web]  | [UPC Image Processing Group][gpi-web] |  [National Institute of Informatics][nii-web] | 

[upc-web]: http://www.upc.edu/?set_language=en 
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 
[nii-web]: http://www.nii.ac.jp/en/

[logo-upc]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/logos/upc.jpg "Universitat Politecnica de Catalunya (UPC)"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/logos/gpi.png "UPC Image Processing Group"
[logo-nii]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/logos/nii.png "National Institute of Informatics"

## Publication
### Abstract

Image representations derived from pre-trained Convolutional Neural Networks (CNNs) have become the new state of the art in computer vision tasks such as instance retrieval. This work explores the suitability for instance retrieval of image- and region-wise representations pooled from an object detection CNN such as Faster R-CNN. We take advantage of the object proposals learned by a Region Proposal Network (RPN) and their associated CNN features to build an instance search pipeline composed of a first filtering stage followed by a spatial reranking. We further investigate the suitability of Faster R-CNN features when the network is fine-tuned for the same objects one wants to retrieve. We assess the performance of our proposed system with the Oxford Buildings 5k, Paris Buildings 6k and a subset of TRECVid Instance Search 2013, achieving competitive results.

### Cite

Our [preprint](http://arxiv.org/abs/1604.08893) is publicly available on arXiv. 

Please cite with the following Bibtex code:

````
@inproceedings{salvador2016faster,
  title={Faster R-CNN Features for Instance Search},
  author={Salvador, Amaia and Giro-i-Nieto, Xavier and Marques, Ferran and Satoh, Shin'ichi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2016}
}
````

![Image of the paper](https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/figs/paper.jpg)

You may also want to refer to our publication with the more human-friendly Chicago style:

````
Amaia Salvador, Xavier Giro-i-Nieto, Ferran Marques and Shin'ichi Satoh. "Faster R-CNN Features for Instance Search." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2016.
````

### Talk on video

<iframe src="https://player.vimeo.com/video/165478041" width="640" height="480" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
<p><a href="https://vimeo.com/165478041">2016-05-Seminar-AmaiaSalvador-DeepVision</a> from <a href="https://vimeo.com/gpi">Image Processing Group</a> on <a href="https://vimeo.com">Vimeo</a>.</p>

### Slides

<iframe src="//www.slideshare.net/slideshow/embed_code/key/lZzb4HdY6OEZ01" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/convolutional-features-for-instance-search" title="Convolutional Features for Instance Search" target="_blank">Convolutional Features for Instance Search</a> </strong> from <strong><a href="//www.slideshare.net/xavigiro" target="_blank">Xavier Giro</a></strong> </div>

## Code Instructions

This python repository contains the necessary tools to reproduce the retrieval pipeline based on off-the-shelf Faster R-CNN features.

### Setup

- You need to download and install Faster R-CNN [python implementation by Ross Girshick](https://github.com/rbgirshick/py-faster-rcnn). Point ```params['fast_rcnn_path']``` to the Faster R-CNN root path in ```params.py```.
- Download [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) Buildings datasets. There are scripts under ```data/images/paris``` and ```data/images/oxford/``` that will do that for you.
- Download Faster R-CNN models by running ```data/models/fetch_models.sh```.

### Usage

- Data preparation. Run ```read_data.py``` to create the lists of query and database images. Run this twice changing ```params['dataset']``` to ```'oxford'``` and ```'paris'```.
- Feature Extraction. Run ```features.py``` to extract Fast R-CNN features for all images in a dataset and store them to disk.
- Ranking. Run ```ranker.py``` to generate and store the rankings for the queries of the chosen dataset.
- Rerank based on region features by running ```rerank.py```.
- Evaluation. Run ```eval.py``` to obtain the Average Precision.
- Visualization. Run ```vis.py```to populate ```data/figures``` with the visualization of the top generated rankings for each query.

## Behind the scenes

![gpi-photo]

[gpi-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master//figs/gpi-small.jpg "Amaia Salvador at the Universitat Politecnica de Catalunya 2016"

## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/authors/gil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-deepvision/master/authors/pujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the project [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 

[logo-nvidia]: ./logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: ./logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: ./logos/MEyC.png "Logo of Spanish government"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the public issues section on this github repo. Alternatively, drop us an e-mail at amaia.salvador@upc.edu or xavier.giro@upc.edu.
