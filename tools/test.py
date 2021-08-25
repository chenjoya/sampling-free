import argparse, os

import torch

from sampling_free.config import cfg
from sampling_free.data import make_data_loader
from sampling_free.engine import do_inference
from sampling_free.modeling import build_model
from sampling_free.utils import Checkpointer, setup_logger, mkdir

def eval_checkpoint(cfg, model, output_dir, num_gpus):
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    for idx, dataset_name in enumerate(dataset_names):
        output_folder = os.path.join(output_dir, dataset_name)
        mkdir(output_folder)
        output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=num_gpus>1)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        model.eval()
        results = do_inference(
            model,
            data_loader_val,
            iou_types=iou_types,
            output_folder=output_folder,
        )

def main():
    parser = argparse.ArgumentParser(description="sampling-free")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.num_gpus = int(os.environ["WORLD_SIZE"])
    args.device_id = int(os.environ["LOCAL_RANK"])
 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if args.num_gpus > 1:
        torch.cuda.set_device(args.device_id)
        torch.distributed.init_process_group(backend="nccl")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = args.config_file.replace("config.yaml", "inference")
    if output_dir:
        mkdir(output_dir)
    
    logger = setup_logger("sampling-free", output_dir, args.device_id)
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    from torch.utils.collect_env import get_pretty_env_info
    logger.info("\n" + get_pretty_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg).cuda(args.device_id)
    
    if args.num_gpus > 1:
        logger.info("Use PyTorch DDP inference")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.device_id]
        )
    
    _ = Checkpointer(cfg, model)
        
    eval_checkpoint(cfg, model, output_dir, args.num_gpus)

if __name__ == "__main__":
    main()
