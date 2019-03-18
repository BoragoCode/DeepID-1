from config import configer, updateConfiger
from train import train_classify_only, train_classify_with_verify
from test import test_classify_only

def main():

    # for patch in range(9):

    #     for scale in ['S', 'M', 'L']:

    #         configer.patch = patch
    #         configer.scale = scale

    #         updateConfiger(configer)

    #         if not configer.with_verify:
    #             ## train classify models
    #             train_classify_only(configer)
    #         else:
    #             ## with verify
    #             train_classify_with_verify(configer)


    for patch in range(9):

        for scale in ['S', 'M', 'L']:

            configer.patch = patch
            configer.scale = scale

            updateConfiger(configer)

            test_classify_only(configer, True)

if __name__ == "__main__":
    main()