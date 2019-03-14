from config import configer
from train import train_classify_only, train_classify_with_verify

def main():

    ## train classify models
    train_classify_only(configer)

    ## with verify
    train_classify_with_verify(configer)


if __name__ == "__main__":
    main()