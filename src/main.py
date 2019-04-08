from config import configer, updateConfiger
from train import *
from test  import *


def train_classifiers(configer):

    ## train 27 models! 

    for patch in range(3, 9):

        for scale in ['S', 'M', 'L']:

            configer.patch = patch
            configer.scale = scale

            updateConfiger(configer)

            train_classify_only(configer)

def train_deepid(configer):

    train_deepid_net(configer)

def main():

    # train_classifiers(configer)
    # train_classify_only(configer)

    train_deepid_net(configer)
    # test_deepid_net(configer)


if __name__ == "__main__":
    main()
