# ShuffleNetV1

# Running on AiQu

After starting the job, clone the repo to the root (same directory where project and data directories exist) directory. Once the repo is cloned, run the following commands

````bash
python3 divide_boat_no_boat.py --src-dir ../data/train/images --csv ../data/train_ship_segmentations_v2.csv --dest-dir ./data
````
````bash
python3 train.py --save-dir ../project/shuffleNet-results --save-interval 5 --img-height 768 --img-width 768 --num-classes 2 --groups 3 --batch-size 100 --epochs 1
````

````bash
python3 train_alt.py --save-dir ../project/base_model_results --img-height 768 --img-width 768 --num-classes 2 --batch-size 100 --epochs 1
````
