"""
Need cv2 installed to run this script
Usage: --imgDir [imgdir] --w 90 --h 60
"""
import os
import sys
import xml.etree.ElementTree as ET
import argparse
import cv2
from shutil import rmtree as rm_dir
from functools import reduce

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--imgDir', help='Images directory recursive')
    parser.add_argument('--w', default=90, help="Image's width after resize")
    parser.add_argument('--h', default=60, help="Image's height after resize")

    args = parser.parse_args()
    dirname = os.getcwd()
    imgdir = args.imgDir
    img_w = int(args.w)
    img_h = int(args.h)

    assert os.path.exists(imgdir), 'Error! imgdir is not correct.'

    if imgdir[-1] is '\\' or imgdir[-1] is '/':
        imgdir = imgdir[:-1]
    absimgdir = dirname + '/' + imgdir
    outputDir = imgdir + "_{}x{}".format(img_w, img_h)
    if os.path.exists(outputDir):
        rm_dir(outputDir)
    os.mkdir(outputDir)
    filenames = [[os.path.join(os.path.split(root)[-1], f) for f in files] for root, _, files in os.walk(imgdir)]
    filenames = reduce(concat, filenames)
    # files_name_only = [files for _, _, files in os.walk(imgdir)]
    input_names = [os.path.join(imgdir, f) for f in filenames]
    output_names = [os.path.join(outputDir, f) for f in filenames]

    files = [os.path.join(imgdir, filename) for filename in filenames if os.path.isfile(os.path.join(imgdir, filename))]


    print("Starting............")
    print("--------------------")

    for idx, filename in enumerate(input_names):
        im_resize = resize(filename, img_w, img_h)
        save_recursive(im_resize, output_names[idx])
    print("Done")

def concat(arr_1, arr_2):
    return arr_1 + arr_2

def resize(filename, size_w, size_h):
    im = cv2.imread(filename)
    im_resize = cv2.resize(im, (size_w, size_h))
    return im_resize

def save(image, filename, where_to_save):
    filename = os.path.join(where_to_save, filename)
    cv2.imwrite(filename, image)

def save_recursive(image, files_to_save):
    folder, _ = os.path.split(files_to_save)
    if not os.path.exists(folder):
        os.mkdir(folder)
    cv2.imwrite(files_to_save, image)

if __name__ == "__main__":
    main()