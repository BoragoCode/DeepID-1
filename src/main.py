from config import configer
from train import train_classify_only, train_classify_with_verify

def main():
    if not configer.with_verify:
        ## train classify models
        train_classify_only(configer)
    else:
        ## with verify
        train_classify_with_verify(configer)


if __name__ == "__main__":
    main()