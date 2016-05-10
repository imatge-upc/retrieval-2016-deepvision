import sys, os
import cv2
import time
import numpy as np
from params import get_params
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

params = get_params()

# Add Faster R-CNN module to pythonpath
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
from fast_rcnn.config import cfg
import test as test_ops


def learn_transform(params,feats):
    
    normalize(feats)
    pca = PCA(params['dimension'],whiten=True)
    
    pca.fit(feats)
        
    pickle.dump(pca,open(params['pca_model'] + '_' + params['dataset'] + '.pkl','wb'))
    

class Extractor():

    def __init__(self,params):

        self.dimension = params['dimension']
        self.dataset = params['dataset']
        self.pooling = params['pooling']
        # Read image lists
        with open(params['query_list'],'r') as f:
            self.query_names = f.read().splitlines()

        with open(params['frame_list'],'r') as f:
            self.database_list = f.read().splitlines()

        # Parameters needed
        self.layer = params['layer']
        self.save_db_feats = params['database_feats']

        # Init network
        if params['gpu']:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()
        print "Extracting from:", params['net_proto']
        cfg.TEST.HAS_RPN = True
        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)

    def extract_feat_image(self,image):

        im = cv2.imread(image)
        
        scores, boxes = test_ops.im_detect(self.net, im, boxes = None)
        feat = self.net.blobs[self.layer].data

        return feat
        

    def pool_feats(self,feat):
        
        if self.pooling is 'max':
            
            feat = np.max(np.max(feat,axis=2),axis=1)
        else:
            
            feat = np.sum(np.sum(feat,axis=2),axis=1)
            
        return feat
        
    def save_feats_to_disk(self):
                
        print "Extracting database features..."
        t0 = time.time()
        counter = 0

        # Init empty np array to store all databsae features
        xfeats = np.zeros((len(self.database_list),self.dimension))
        
        for frame in self.database_list:
            counter +=1
            
            # Extract raw feature from cnn
            feat = self.extract_feat_image(frame).squeeze()
            
            # Compose single feature vector
            feat = self.pool_feats(feat)
            
            # Add to the array of features
            xfeats[counter-1,:] = feat
            
            # Display every now and then
            if counter%50 == 0:
                print counter, '/', len(self.database_list), time.time() - t0
        
        
        # Dump to disk
        pickle.dump(xfeats,open(self.save_db_feats,'wb'))
        
        print " ============================ "
        

if __name__ == "__main__":
    params = get_params()
    
    E = Extractor(params)
    E.save_feats_to_disk()
    
    feats = pickle.load(open(params['database_feats'],'rb'))
    learn_transform(params,feats)
