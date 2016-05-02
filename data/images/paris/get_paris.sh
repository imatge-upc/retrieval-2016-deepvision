# Get data
mkdir data/
curl http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz | tar xz -C data/
curl http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz | tar xz -C data/

# Get labels
mkdir groundtruth/
curl http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz | tar xz -C groundtruth/
