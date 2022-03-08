mkdir data
python3 divide_boat_no_boat.py --sourcedir ../data/train/images --csv ../data/train_ship_segmentations_v2.csv --destdir ./data
python3 trian --batch-size 128 --epochs 100 --save ../project/shuffleNet-results/models 