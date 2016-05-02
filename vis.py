import os,pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2, random
import Image
from params import get_params
from eval import Evaluator

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


class Visualization():
    
    def __init__(self,params):
        
        self.dataset= params['dataset']
        self.image_path = params['database_images']
        self.class_scores = params['use_class_scores']
       
        self.queries = params['query_names']   
        self.rankings_dir = params['rankings_dir']
        
        self.size_box = params['size_box']
        self.stage = params['stage']
        self.N_display = params['N_display']
        self.figsize = params['figsize']
        self.figures_path = params['figures_path']
        
        self.reranking_path = params['reranking_path']        
        
        with open(params['query_list'],'r') as f:
            self.query_names = f.read().splitlines()
                  
        self.ground_truth = params['ground_truth_file']
    
        
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
    
    def get_query_im(self,query):
            
        query,bbx = self.query_info(query)
        im = cv2.imread(query)
            
        
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        cv2.rectangle(im, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), (255,0,0),self.size_box)
        
        return im
        
    def create_thumb(self,im):
    
        x = 800
        y = 800
        size = (y,x)
        image = Image.fromarray(im)
        
        image.thumbnail(size, Image.ANTIALIAS)
        background = Image.new('RGBA', size, "black")
        background.paste(image, ((size[0] - image.size[0]) / 2, (size[1] - image.size[1]) / 2))
        
        return np.array(background)[:,:,0:3]
    
    def vis_one_query(self,query,ranking):
        
        grid_size_x = self.N_display + 1
        grid_size_y = 1
        pos_in_fig = 1
        
        fig = plt.figure(figsize=self.figsize)
        
        ax = fig.add_subplot(grid_size_y, grid_size_x, pos_in_fig)
        
        query_im = self.get_query_im(query)
        query_im = self.create_thumb(query_im)
        query_im = cv2.copyMakeBorder(query_im,30,30,30,30,cv2.BORDER_CONSTANT,value= [0,0,255])
        
        ax.imshow(query_im)
        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        ranking = self.read_ranking(query)

        junk = np.loadtxt(os.path.join(self.ground_truth,os.path.basename(query).split('_query.txt')[0] + '_junk.txt'),dtype="str")
        ok = np.loadtxt(os.path.join(self.ground_truth,os.path.basename(query).split('_query.txt')[0] + '_ok.txt'),dtype = "str")
        good = np.loadtxt(os.path.join(self.ground_truth,os.path.basename(query).split('_query.txt')[0] + '_good.txt'),dtype = "str")
        
        if self.stage is 'rerank':
            
            with open(os.path.join(self.reranking_path,os.path.basename(query.split('_query')[0]) + '.pkl') ,'rb') as f:
                distances = pickle.load(f)
                locations = pickle.load(f)
                frames = pickle.load(f)
                class_ids = pickle.load(f)
            
            if self.class_scores:
                
                frames_sorted = np.array(frames)[np.argsort(distances)[::-1]]
                locations_sorted = np.array(locations)[np.argsort(distances)[::-1]]
            
            else:
                frames_sorted = np.array(frames)[np.argsort(distances)]
                locations_sorted = np.array(locations)[np.argsort(distances)]
            
        for i in range(self.N_display):

            frame = ranking[i]
            
            if self.dataset is 'paris':
                
                frame_to_read = os.path.join(self.image_path,frame.split('_')[1],frame + '.jpg')
                
            elif self.dataset is 'oxford':
                
                frame_to_read = os.path.join(self.image_path,frame + '.jpg')
                
            
            im = cv2.imread(frame_to_read)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            
            if self.stage is 'rerank':
                
                # paint box too
                bbx = locations_sorted[i,:]
                print bbx
                cv2.rectangle(im, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), (255,0,0),self.size_box)

            im = self.create_thumb(im)
            
                
            if os.path.basename(ranking[i]).split('.jpg')[0] in good:
                # GREEN
                im = cv2.copyMakeBorder(im,30,30,30,30,cv2.BORDER_CONSTANT,value= [0,255,0])
        
            elif os.path.basename(ranking[i]).split('.jpg')[0] in ok:
                # Yellow
                im = cv2.copyMakeBorder(im,30,30,30,30,cv2.BORDER_CONSTANT,value= [0,255,0])
        
            elif os.path.basename(ranking[i]).split('.jpg')[0] in junk:
                # ORANGE
                im = cv2.copyMakeBorder(im,30,30,30,30,cv2.BORDER_CONSTANT,value= [0,255,0])
            else:
                # RED
                im = cv2.copyMakeBorder(im,30,30,30,30,cv2.BORDER_CONSTANT,value= [255,0,0])
           

            ax2 = fig.add_subplot(grid_size_y, grid_size_x, pos_in_fig + i+1)
    
            ax2.imshow(im)
            
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            
            '''
            if self.stage is 'rerank' and not self.ft_network:
                ax2.set_title(CLASSES[class_ids[i]], fontsize=50)
            '''
            fig.tight_layout()
    
        fig.savefig(os.path.join(self.figures_path,os.path.basename(query).split('_query')[0] + '.png'))
        plt.close()
        
    def vis(self):
        
        iter_ = self.query_names
        
        for query in iter_:
            print query
            ranking = self.read_ranking(query)
            self.vis_one_query(query,ranking)
    
if __name__ == '__main__':
        
    params = get_params()
    V = Visualization(params)
    V.vis()