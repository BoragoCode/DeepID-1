from gen_detect import *
from gen_classify_samples import *
from gen_verify_samples import *
from gen_deepid_pairs import *

def main():
    datadir = '/home/louishsu/Work/Codes/DeepID/data/lfw-deepfunneled'

    ## step 1. detect all images
    # detect_lfw(datadir)

    ## step 2. generate label file
    # gen_labels(datadir)

    ## step 3. generate classify samples
    # for i in range(9):
    #     gen_classify(datadir, i, 1.2)

    ## step 4. generate verify samples
    # gen_verify_pairs(datadir)

    ## step 5. generate finetune samples
    # for i in range(9):
    #     gen_classify_similarity_pairs(datadir, i, ratio=1.2)

    gen_deepid_pair_samples(datadir)
    
if __name__ == "__main__":
    main()