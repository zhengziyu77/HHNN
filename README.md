# Defining and Discovering Hyper-meta-paths for Heterogeneous Hypergraphs(NeurIPS 2025)

## Dataset
Please unzip dataset.zip to data folder(Movielens and Olist dataset)


## Run Experiment
python train.py --method HyperHIN  --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 128 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 --cuda 0   --train_prop 0.6 --type_att_size 32 --type_fusion 'att' --use_node_attn 'att' --use_hyperedge_attn 'att'


