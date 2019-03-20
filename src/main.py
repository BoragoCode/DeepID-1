from config import configer, updateConfiger
from train import train_classify_only, train_verify, train_classify_with_similarity, train_deepid_net
from test import test_classify_only, test_verify, test_classify_with_similarity, test_deepid_net

def main():

    ## train 27 models! 
    # for patch in range(9):

    #     for scale in ['S', 'M', 'L']:

    #         configer.patch = patch
    #         configer.scale = scale

    #         updateConfiger(configer)

    #         train_classify_only(configer)



    ## finetune model using combined loss
    # for patch in range(9):

    #     for scale in ['S', 'M', 'L']:

    #         configer.patch = patch
    #         configer.scale = scale

    #         updateConfiger(configer)

    #         train_classify_with_similarity(configer)



    ## get features to train verify model
    # for patch in range(9):

    #     for scale in ['S', 'M', 'L']:

    #         configer.patch = patch
    #         configer.scale = scale

    #         updateConfiger(configer)

    #         test_classify_only(configer, True)



    ## train verify model
    # train_verify(configer)
    # test_verify(configer)




    ## train whole net
    # train_deepid_net(configer)
    test_deepid_net(configer)


if __name__ == "__main__":
    main()