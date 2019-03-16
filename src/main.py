from config import configer, updateConfiger
from train import train_classify_only, train_classify_with_verify

def main():

    for patch in range(2, 9):

        for scale in ['S', 'M', 'L']:

            configer.patch = patch
            configer.scale = scale

            updateConfiger(configer)

            if not configer.with_verify:
                ## train classify models
                train_classify_only(configer)
            else:
                ## with verify
                train_classify_with_verify(configer)


if __name__ == "__main__":
    main()