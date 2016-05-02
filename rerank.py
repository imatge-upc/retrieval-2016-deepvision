import sys, os,cv2, time, random
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import math
from params import get_params
import pickle
params = get_params()
from sklearn.preprocessing import normalize

# Add Faster R-CNN module to pythonpath
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
import test as test_ops

class Reranker():
    
    def __init__(self,params):
        
        self.dataset= params['dataset']
        self.image_path = params['database_images']
        self.dimension = params['dimension']
        self.layer = params['layer']
        self.top_n = params['num_rerank']
        self.reranking_path = params['reranking_path']
        self.REG_BOXES = params['use_regressed_boxes']
        self.pooling = params['pooling']
        self.stage = params['stage']
        self.N_QE = params['N_QE']
        self.class_scores = params['use_class_scores']
           
        with open(params['frame_list'],'r') as f:
            self.database_list = f.read().splitlines()  
      
        with open(params['query_list'],'r') as f:
            self.query_names = f.read().splitlines()
      
        # Distance type
        self.dist_type = params['distance']
        
        # Where to store the rankings
        self.rankings_dir = params['rankings_dir']

        # Init network
        if params['gpu']:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
        self.queries = params['query_names']
        # List of queries
        
        
        if self.pooling is 'sum':
            # PCA Models
            if self.dataset is 'paris':
                
                self.pca = pickle.load(open(params['pca_model'] + '_oxford.pkl', 'rb'))
                
            elif self.dataset is 'oxford':
                
                self.pca = pickle.load(open(params['pca_model'] + '_paris.pkl', 'rb'))
 
    def extract_feat_image(self,image):

        im = cv2.imread(image)
        
        scores, boxes = test_ops.im_detect(self.net, im, boxes = None,REG_BOXES=self.REG_BOXES)
        
  
        layer_roi = 'pool5'
        feat = self.net.blobs[layer_roi].data
        

        return feat,boxes,scores
    
    def read_ranking(self,query):
        
        
        with open(os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt'),'r') as f:
            ranking = f.read().splitlines()
            
        
        return ranking
    
    def query_info(self,filename):
        
        '''
        For oxford and paris, get query frame and box 
        '''

        data = np.loadtxt(filename, dtype="str")
        
        if self.dataset is 'paris':
            
            query = data[0]
                
        elif self.dataset is 'oxford':
                
            query = data[0].split('oxc1_')[1]
        
        bbx = data[1:].astype(float).astype(int)
        
        if self.dataset is 'paris':
            query = os.path.join(self.image_path,query.split('_')[1],query + '.jpg')
        elif self.dataset is 'oxford':
            query = os.path.join(self.image_path,query + '.jpg')
    
        return query, bbx 
    
    def get_query_local_feat(self,query,box=None):
        
        '''
        Extract local query feature using bbx
        '''
        if box is None:
            
            # For paris and oxford
            query,bbx = self.query_info(query)
            
        else:
            
            # locations are provided
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            
        im = cv2.imread(query)
        
        height = np.shape(im)[0]
        width = np.shape(im)[1]
    
        
        # Forward pass
        scores, boxes = test_ops.im_detect(self.net, im, boxes = None)
        
        # Get conv5 layer
        feat = self.net.blobs[self.layer].data.squeeze()
                
        # Get the image/feature ratio
        mult_h = float(np.shape(feat)[1])/height
        mult_w = float(np.shape(feat)[2])/width
    
        # Resize the bounding box to feature size
        if box is None:
            
            # Adjust query bounding box to feature space
            bbx[0] *= mult_w
            bbx[2] *= mult_w
            bbx[1] *= mult_h
            bbx[3] *= mult_h
            
        else:
            
            bbx = [int(math.floor(xmin*mult_w)),int(math.floor(ymin*mult_h)),int(math.ceil(xmax*mult_w)),int(math.ceil(ymax*mult_h))]
            
        # Crop local features with bounding box
        local_feat = feat[:,bbx[1]:bbx[3],bbx[0]:bbx[2]]
        
        # sum pool
        if self.pooling is 'sum':
            local_feat =  np.sum(np.sum(local_feat,axis=1),axis=1)
        else:
            local_feat =  np.max(np.max(local_feat,axis=1),axis=1) 
            
        return local_feat
    
    def rerank_one_query(self,query,num_queries):
        
        # Init query feat vector
        query_feats = np.zeros((self.dimension))
        for i in np.arange(num_queries)+1:

            query_ = query
            query_name = os.path.basename(query).rsplit('_',2)[0]
            
            # Generate query feature and add it to matrix
            query_feats += self.get_query_local_feat(query_)
        
        query_feats/=num_queries
        
        
        if self.stage is 'rerank2nd':
            # second stage of reranking. taking N locations at top N ranking as queries...

            with open(os.path.join(self.reranking_path,os.path.basename(query.split('_query')[0]) + '.pkl') ,'rb') as f:
                distances = pickle.load(f)
                locations = pickle.load(f)
                frames = pickle.load(f)
                class_ids = pickle.load(f)
                
            frames_sorted = np.array(frames)[np.argsort(distances)]
            locations_sorted = np.array(locations)[np.argsort(distances)]
            
            for i_qe in range(self.N_QE):
                query_feats +=self.get_query_local_feat(frames_sorted[i_qe],locations_sorted[i_qe])
            
            query_feats/=(self.N_QE+1)
        
        query_feats = query_feats.T
        
        normalize(query_feats)
        
        
        if self.pooling is 'sum':
            # Apply PCA
            query_feats = self.pca.transform(query_feats)
            
            normalize(query_feats)
        
        # Read baseline ranking   
        ranking = self.read_ranking(query)
        
        # Rerank
        distances,locations, frames,class_ids = self.rerank_top_n(query_feats,ranking,query_name)
        
        with open(os.path.join(self.reranking_path,os.path.basename(query.split('_query')[0]) + '.pkl') ,'wb') as f:
            pickle.dump(distances,f)
            pickle.dump(locations,f)
            pickle.dump(frames,f)
            pickle.dump(class_ids,f)
        # Write new ranking to disk
        self.write_rankings(query,ranking,distances)
        
    def rerank_top_n(self,query_feats,ranking,query_name):
        
        distances = []
        locations = []
        frames = []
        class_ids = []
        #query_feats = query_feats.T
        
        # query class (+1 because class 0 is the background)
        cls_ind = np.where(np.array(self.queries) == str(query_name))[0][0] + 1
        
        for im_ in ranking[0:self.top_n]:
            
            if self.dataset is 'paris':
                frame_to_read = os.path.join(self.image_path,im_.split('_')[1],im_ + '.jpg')
            elif self.dataset is 'oxford':
                frame_to_read = os.path.join(self.image_path,im_ + '.jpg')
            
            frames.append(frame_to_read)
            # Get features of current element
            feats,boxes,scores = self.extract_feat_image(frame_to_read)
            
            # we rank based on class scores 
            if self.class_scores:
                
                scores = feats[:,cls_ind]
                
                # position with highest score for that class
                best_pos = np.argmax(scores)
                
                # array of boxes with higher score for that class
                best_box_array = boxes[best_pos,:]
                
                # single box with max score for query class
                best_box = best_box_array[4*cls_ind:4*(cls_ind + 1)]
                
                # the actual score
                distances.append(np.max(scores))
                locations.append(best_box)
                class_ids.append(cls_ind)
                
            else:
                
                
                if self.pooling is 'sum':
                    # pca transform
                    feats = np.sum(np.sum(feats,axis=2),axis=2)
                    normalize(feats)
                    feats = self.pca.transform(feats)
                    normalize(feats)
                else:
                    feats = np.max(np.max(feats,axis=2),axis=2)
                    normalize(feats)
                
                # Compute distances
                dist_array = pairwise_distances(query_feats,feats,self.dist_type, n_jobs=-1)
                
                
                # Select minimum distance
                distances.append(np.min(dist_array))
                
                # Array of boxes with min distance
                idx = np.argmin(dist_array)
                
                # Select array of locations with minimum distance
                best_box_array = boxes[idx,:]
                
                # Discard background score
                scores = scores[:,1:]
                
                # Class ID with max score . 
                cls_ind = np.argmax(scores[idx,:]) 
                class_ids.append(cls_ind+1)
            
                # Select the best box for the best class
                best_box = best_box_array[4*cls_ind:4*(cls_ind + 1)]
                            
                locations.append(best_box)
                
        return distances,locations, frames, class_ids
          
    def rerank(self):

        iter_ = self.query_names
        num_queries = 1
        
        i = 0
        for query in iter_:
            print "Reranking for query", i, "out of", len(iter_), '...'
            i+=1
            self.rerank_one_query(query,num_queries)
                        
    def write_rankings(self,query,ranking,distances):
        
        if self.class_scores:
            new_top_r = list(np.array(ranking[0:self.top_n])[np.argsort(distances)[::-1]])
        else:
            new_top_r = list(np.array(ranking[0:self.top_n])[np.argsort(distances)])
        
        ranking[0:self.top_n] = new_top_r
        

        savefile = open(os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt'),'w')
                
        for res in ranking:
    
            savefile.write(os.path.basename(res).split('.jpg')[0] + '\n')
    
        savefile.close()
            
        
    
if __name__== '__main__':
    
    params = get_params()
    
    RR = Reranker(params)
    
    RR.rerank()
        
    
        
        
            
            
            