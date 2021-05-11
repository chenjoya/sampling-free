model=sampling_free/fcos_R_50_FPN_1x

CUDA_VISIBLE_DEVICES=6 python #-m torch.distributed.launch \
#    --nproc_per_node=4 \
#    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file configs/$model\.yaml \
    DATALOADER.NUM_WORKERS 8 \
    OUTPUT_DIR outputs/$model
