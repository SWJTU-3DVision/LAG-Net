CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ../market1501/ --batchid 8 --batchtest 4 --test_every 50 --epochs 600 --decay_type step_320_380_420 --loss 1*CrossEntropy+2*Triplet+1*L2 --margin 1.2 --nGPU 1  --lr 2e-4 --optimizer ADAM --random_erasing --amsgrad --save lagnet


