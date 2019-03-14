from config import configer
from train import train_classify_only

def main():

    ## train classify models
    train_classify_only(configer)

if __name__ == "__main__":
    main()