import os
import sys
import numpy as np
from shutil import rmtree
import argparse
from pprint import *

def main(args):
    imgdir = os.path.expanduser(args.imgdir)
    assert os.path.exists(imgdir), imgdir + " is not a valid directory.\n"
    with open(imgdir, "r") as f:
        for line in f:
            print(line)
            break
            # f.readline()
    pass


if __name__ == "__main__":
    handler = argparse.ArgumentParser()
    handler.add_argument("--imgdir", "-d", default="")
    FLAGS = handler.parse_args()
    main(FLAGS)