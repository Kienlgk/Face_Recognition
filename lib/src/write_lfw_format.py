import os
from shutil import rmtree
import argparse
from functools import reduce

tab = "	"
supported_img_ext = [".png", ".jpg", ".bmp", ".jpeg"]

def main():
    handler = argparse.ArgumentParser()
    handler.add_argument("--imgdir", "-d")

    flags = handler.parse_args()
    imgdir  = flags.imgdir
    assert os.path.exists(imgdir)
    folders = os.listdir(imgdir)
    # print(folders)
    write_pairs_file(imgdir, folders)
    print("Done.")


def get_l1_children_folders_from_path(super_path):
    return [child for child in os.listdir(super_path)
            if os.path.exists(os.path.join(super_path, child)) and not os.path.isfile(os.path.join(super_path, child))]

def write_pairs_file(basedir, folder_list):
    with open("generated_pairs.txt", "w+") as pairs_f:
        print("10	300", file=pairs_f)
        for folder in folder_list:
            noImg = len(os.listdir(os.path.join(basedir, folder)))
            print("{}\t{}\t{}".format(folder, str(1), str(noImg)), file=pairs_f)

main()