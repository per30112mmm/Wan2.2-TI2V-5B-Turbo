GPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=${GPUS} --nnodes=${NODE_COUNT} --node_rank="${RANK}" --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py \
    --config_path configs/self_forcing_wan22_dmd.yaml \
    --logdir $LOGDIR \
    --data_path data/MagicData.csv \
    --no_visualize \
    --disable-wandb