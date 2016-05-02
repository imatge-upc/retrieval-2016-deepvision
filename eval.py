import numpy as np
import pickle, os
from params import get_params

class Evaluator():

    def __init__(self,params):
        self.dataset = params['dataset']
        self.rankings_dir = params['rankings_dir']
        self.ground_truth = params['ground_truth_file']
        self.query_list = params['query_names']
        self.K = params['K']

    def read_ranking(self,query):

        ranking = pickle.load(open(os.path.join(self.rankings_dir,query + '.pkl'),'rb'))

        return ranking

    def read_ground_truth(self):

        self.gt = np.loadtxt( self.ground_truth, dtype='string' )

    def relnotrel(self,query_name,ranking):

        # Extract shots for the query
        query_shots = self.gt[ (self.gt[:,0] == query_name)]

        # Extract relevant shots for the query
        rel_shots = query_shots[query_shots[:,3] == '1']

        # Total Number of relevant shots in the ground truth
        total_relevant = np.shape(rel_shots)[0]

        relist = np.zeros((1, len(ranking)))

        i = 0
        for shot in ranking:

            if shot in rel_shots:
                relist[0,i] = 1

            i +=1

        return relist.squeeze(), total_relevant

    def average_precision(self,relist,total_relevant):

        accu = 0
        num_rel = 0

        for k in range(min(len(relist),self.K)):

            if relist[k] == 1:
                num_rel+=1

                accu += float(num_rel)/float(k+1)

        return accu/total_relevant

    def run_evaluation(self):
        
        if self.dataset is 'trecvid':
            self.read_ground_truth()
    
            ap_list = []
            for query in self.query_list:
    
                ranking = self.read_ranking(query)
                relist, total_relevant = self.relnotrel(query,ranking)
                ap = self.average_precision(relist,total_relevant)
                ap_list.append(ap)
        else:
            dic_res = {}
            ap_list = []
            
            for q_name in self.query_list:
                print q_name
                for i in range(1,6 ): # 6 images per query
                    cmd = "./compute_ap {}_{} {}/{}_{}.txt > tmp.txt".format( os.path.join(self.ground_truth, q_name), i, self.rankings_dir, q_name, i)
                    os.system( cmd )
                    ap = np.loadtxt("tmp.txt")
               	    dic_res[q_name+"_"+str(i)]=ap
                    print ap
                    ap_list.append(ap)
            
        return ap_list

if __name__ == "__main__":

    params = get_params()
    E = Evaluator(params)
    ap_list = E.run_evaluation()
    
    for ap in ap_list:
        print ap
    print "====="
    print np.mean(ap_list)
