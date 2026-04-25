# MiniImageNet
# MAML
python main.py --dataset mini-imagenet --algorithm MAML --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset mini-imagenet --algorithm MAML --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

# FOMAML (alternatively, one can also use --algorithm MAML --first-order)
python main.py --dataset mini-imagenet --algorithm FOMAML --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset mini-imagenet --algorithm FOMAML --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

# TruncMAML (change the trunc argument as desired)
python main.py --dataset mini-imagenet --algorithm TruncMAML --trunc 1 --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset mini-imagenet --algorithm TruncMAML --trunc 1 --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

# BinomMAML (change the trunc argument as desired)
python main.py --dataset mini-imagenet --algorithm BinomMAML --trunc 1 --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset mini-imagenet --algorithm BinomMAML --trunc 1 --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

# TieredImageNet
# MAML
python main.py --dataset tiered-imagenet --algorithm MAML --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset tiered-imagenet --algorithm MAML --num-cls 5 --num-trn-data 5  --meta-lr 0.001 --task-lr 0.01

# FOMAML (alternatively, one can also use --algorithm MAML --first-order)
python main.py --dataset tiered-imagenet --algorithm FOMAML --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset tiered-imagenet --algorithm FOMAML --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

# TruncMAML (change the trunc argument as desired)
python main.py --dataset tiered-imagenet --algorithm TruncMAML --trunc 1 --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset tiered-imagenet --algorithm TruncMAML --trunc 1 --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

# BinomMAML (change the trunc argument as desired)
python main.py --dataset tiered-imagenet --algorithm BinomMAML --trunc 1 --num-cls 5 --num-trn-data 1 --meta-lr 0.001 --task-lr 0.01
python main.py --dataset tiered-imagenet --algorithm BinomMAML --trunc 1 --num-cls 5 --num-trn-data 5 --meta-lr 0.001 --task-lr 0.01

nohup python main.py --dataset tiered-imagenet --algorithm iMAML --num-cls 5 --num-trn-data 1 --trunc 1 --meta-lr 5e-4 &> logs/tiered-imagenet/iMAML-5way1shot-1trunc.log &
nohup python main.py --dataset tiered-imagenet --algorithm iMAML --num-cls 5 --num-trn-data 5 --trunc 1 --meta-lr 5e-4 &> logs/tiered-imagenet/iMAML-5way5shot-1trunc.log &
