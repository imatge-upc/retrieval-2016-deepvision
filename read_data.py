import glob,os
from params import get_params

def create_db_list(PATH,key):

    print "Listing directory:", PATH

    result = [y for x in os.walk(PATH) for y in glob.glob(os.path.join(x[0], key))]

    return result

def create_shot_list(PATH):
    
    shots = os.listdir(PATH)
    
    return shots

if __name__ == "__main__":

    params = get_params()

        
    frames = create_db_list(params['database_images'],'*.jpg')
    
    new_frames = []
    if params['dataset'] is 'paris':
        #remove corrupt images
        for frame in frames:
            if not os.path.basename(frame) in params['paris_corrupt_list']:
                new_frames.append(frame)
        
        frames = new_frames
    print "Listed", len(frames), 'frames.'
    print "Saving text file:", params['frame_list']
    with open(params['frame_list'],'w') as outfile:
        outfile.write("\n".join(frames))
        
    # queries
    queries = create_db_list(params['ground_truth_file'],"*query.txt")
    print "Listed", len(queries), 'frames.'
    print "Saving text file:", params['query_list']
    with open(params['query_list'],'w') as outfile:
        outfile.write("\n".join(queries))
        
    
