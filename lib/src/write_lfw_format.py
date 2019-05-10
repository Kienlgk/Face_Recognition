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
    # folders = os.listdir(imgdir)
    folders = get_l1_children_folders_from_path(imgdir)
    write_pairs_file(imgdir, folders)
    print("Done.")


def get_l1_children_folders_from_path(super_path):
    return [child for child in os.listdir(super_path)
            if os.path.exists(os.path.join(super_path, child)) and not os.path.isfile(os.path.join(super_path, child))]


def get_l1_children_img_from_path(super_path, supported_img_ext=supported_img_ext):
    return [child for child in os.listdir(super_path)
            if os.path.exists(os.path.join(super_path, child)) 
                and os.path.isfile(os.path.join(super_path, child)) 
                and os.path.splitext(child)[-1] in supported_img_ext]


def write_pairs_file(basedir, folder_list):
    with open("generated_pairs.txt", "w+") as pairs_f:
        print("10	300", file=pairs_f)
        write_pairs_file_is_same(basedir, folder_list, pairs_f)
        write_pairs_file_is_not_same(basedir, folder_list, pairs_f)


def write_pairs_file_is_same(basedir, folder_list, log_file):
    for folder in folder_list:
        noImg = len(os.listdir(os.path.join(basedir, folder)))
        for i in range(noImg):
            if i == 0:
                continue
            print("{}\t{}\t{}".format(folder, str(1), str(i+1)), file=log_file)


def write_pairs_file_is_not_same(basedir, folder_list, log_file):
    len_array = [len(get_l1_children_img_from_path(os.path.join(basedir, folder))) for folder in folder_list]
    for folder_idx, folder in enumerate(folder_list):
        if folder_idx == len(folder_list) - 1:
            break
        len_i = len_array[folder_idx]
        len_j = len_array[folder_idx+1]
        for i in range(len_i):
            for j in range(len_j):
                print("{}\t{}\t{}\t{}".format(folder_list[folder_idx], str(i+1), folder_list[folder_idx+1], str(j+1)), file=log_file)
                

main()