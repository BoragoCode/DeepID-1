from gen_detect import *
from gen_labels import *
from gen_classify_samples import *

def main_lfw():
    datadir = '/home/louishsu/Work/Codes/DeepID/data/lfw-deepfunneled'

    ## step 1. detect all images
    detect_lfw(datadir)

    ## step 2. generate label file
    gen_labels(datadir)

    ## step 3. generate classify data
    gen_classify(datadir)

    ## step 4. generate classify pairs
    gen_classify_pairs(datadir)

def main_celeba():
    datadir = '/home/louishsu/Work/Codes/DeepID/data/img_align_celeba_png'

    ## step 1. detect all images
    detect_celeba(datadir)

    ## step 3. generate classify data
    gen_classify_celeba(datadir)

    ## step 4. generate classify pairs
    gen_classify_pairs_celeba(datadir)
    
if __name__ == "__main__":
    # main_lfw()
    main_celeba()