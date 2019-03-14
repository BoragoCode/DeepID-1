from gen_detect import *
from gen_classify_samples import *
from gen_classify_verify_samples import *

def main():
    datadir = '/home/louishsu/Work/Codes/DeepID/data/lfw-deepfunneled'

    ## step 1. detect all images
    # detect_lfw(datadir)

    ## step 2. generate label file
    # gen_labels(datadir)

    ## step 3. generate classify samples
    # for i in range(9):
    #     gen_classify(datadir, i, 1.2)

    # step 4. generate classify with verification samples
    # for i in range(9):
    #     gen_classify_verify_pairs(datadir, i, 1.2)

    
if __name__ == "__main__":
    main()