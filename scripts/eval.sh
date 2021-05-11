model=fcos/fcos_R_50_FPN_1x

CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/$model\.yaml \
    DATALOADER.NUM_WORKERS 8 \
    OUTPUT_DIR outputs/$model
