import argparse
import os
import shutil
from os import listdir
from os.path import isdir, isfile, join

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=str, help='path to the folder containing images')
    parser.add_argument('--csv', type=str, help='your csv file (path)')
    parser.add_argument('--dest-dir', type=str, help='folder where no_boat/boat directory is saved')
    parser.add_argument('--num-imgs', type=int, default=-1, help='Number of images that are to be split. -1 means all of the images will be split')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    # Get target and source directories
    images_path = opt.src_dir

    if not os.path.exists(opt.dest_dir):
        print(f"The destination directory does not exist. Creating a new destination directory...")
        os.makedirs(opt.dest_dir)
        print(f"'{opt.dest_dir}' directory is created!")

    if not os.path.exists(f"{opt.dest_dir}/train/"):
        print(f"The train directory does not exist. Creating a new destination directory...")
        os.makedirs(f"{opt.dest_dir}/train/")
        print(f"'{opt.dest_dir}/train/' directory is created!")

    if not os.path.exists(f"{opt.dest_dir}/test/"):
        print(f"The test directory does not exist. Creating a new destination directory...")
        os.makedirs(f"{opt.dest_dir}/test/")
        print(f"'{opt.dest_dir}/test/' directory is created!")    

    destination_path_boats = f"{opt.dest_dir}/test/boats/"
    destination_path_no_boats = f"{opt.dest_dir}/test/no_boats/"

    if not os.path.exists(destination_path_boats):
        os.makedirs(destination_path_boats)
        print(f"'{destination_path_boats}' directory is created!")
    if not os.path.exists(destination_path_no_boats):
        os.makedirs(destination_path_no_boats)
        print(f"'{destination_path_no_boats}' directory is created!")

    destination_path_boats = f"{opt.dest_dir}/train/boats/"
    destination_path_no_boats = f"{opt.dest_dir}/train/no_boats/"

    if not os.path.exists(destination_path_boats):
        os.makedirs(destination_path_boats)
        print(f"'{destination_path_boats}' directory is created!")
    if not os.path.exists(destination_path_no_boats):
        os.makedirs(destination_path_no_boats)
        print(f"'{destination_path_no_boats}' directory is created!")

    # Load csv-file
    dataset_csv = f"{opt.csv}"

    df = pd.read_csv(dataset_csv)
    print("csv-file loaded:")
    print(df.head())

    # Start copying! 
    print("------------------------------------")
    print("Starting to copy and divide files...")
    max = df.shape[0]

    nr_boats = 0
    nr_no_boats = 0

    for index, row in df.iterrows():
        mask = row["EncodedPixels"]

        img_path = os.path.join(images_path,row['ImageId'])
        img_path_boats = os.path.join(destination_path_boats,row['ImageId'])
        img_path_no_boats = os.path.join(destination_path_no_boats,row['ImageId'])

        # Check boat or not boat
        if os.path.isfile(img_path) and not pd.isna(mask) and not os.path.isfile(img_path_boats):
            shutil.copy(img_path, destination_path_boats)
            nr_boats += 1
        elif os.path.isfile(img_path)and pd.isna(mask) and not os.path.isfile(img_path_no_boats):
            shutil.copy(img_path, destination_path_no_boats)
            nr_no_boats += 1
        
        # Print
        if index % 1000 == 0:
            # clear_output()
            print(f"{index}/{max}")
            print(f"Added number of boats: {nr_boats}")
            print(f"Added number of no boats: {nr_no_boats}")
        
        if opt.num_imgs != -1 and index > opt.num_imgs:
            break
            
    # clear_output()
    print(f"{max}/{max}")
    print(f"Added number of boats: {nr_boats}")
    print(f"Added number of no boats: {nr_no_boats}")
    print("DONE!")

def test_train_split_fn(opt):
    boats_files = listdir(f"{opt.dest_dir}/train/boats/")
    _, test_boats = train_test_split(boats_files, test_size=0.2, shuffle=False)
    no_boats_files = listdir(f"{opt.dest_dir}/train/no_boats/")
    _, test_no_boats = train_test_split(no_boats_files, test_size=0.2, shuffle=False)

    destination_path_boats = f"{opt.dest_dir}/test/boats"
    destination_path_no_boats = f"{opt.dest_dir}/test/no_boats"

    print("Moving boats to test directory...")
    for img in tqdm(test_boats):
        shutil.move(f"{opt.dest_dir}/train/boats/{img}", destination_path_boats)
    
    print("Moving no_boats to test directory...")
    for img in tqdm(test_no_boats):
        shutil.move(f"{opt.dest_dir}/train/no_boats/{img}", destination_path_no_boats)




def delete_duplicates(opt):
    base_path = opt.dest_dir
    test_imgs = listdir(f"{base_path}/test/boats")
    train_imgs = listdir(f"{base_path}/train/boats")

    i = 0
    for img in test_imgs:
        if img in train_imgs:
            if i % 2 == 0:
                os.remove(f"{base_path}/test/boats/{img}")
            else:
                os.remove(f"{base_path}/train/boats/{img}")
            i += 1
            print(f"found {img} in boats")

    test_imgs = listdir(f"{base_path}/test/no_boats")
    train_imgs = listdir(f"{base_path}/train/no_boats")

    i = 0
    for img in test_imgs:
        if img in train_imgs:
            if i % 2 == 0:
                os.remove(f"{base_path}/test/no_boats/{img}")
            else:
                os.remove(f"{base_path}/train/no_boats/{img}")
            i += 1
            print(f"found {img} in no_boats")
            
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
