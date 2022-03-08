mkdir data
python3 divide_boat_no_boat.py --sourcedir ../data/train/images --csv ../data/train_ship_segmentations_v2.csv --destdir ./data
python3 train.py --batch-size 64 --epochs 50 --save ../project/shuffleNet-results/models 