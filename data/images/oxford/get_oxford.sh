# Get data
mkdir data/
curl http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz | tar xz -C data/

# Get labels
mkdir groundtruth/
curl http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz | tar xz -C groundtruth/
