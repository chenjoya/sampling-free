import argparse, os

import torch

from sampling_free.config import cfg
from sampling_free.data import make_data_loader
from sampling_free.engine import do_train
from sampling_free.modeling import build_model
from sampling_free.solver import build_optimizer, build_lr_scheduler
from sampling_free.utils import Checkpointer, setup_logger, mkdir, save_config

def train(cfg, device_id, num_gpus, output_dir, logger):
    model = build_model(cfg).cuda(device_id)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    if num_gpus > 1:
        logger.info("Use PyTorch DDP training")
        if cfg.MODEL.USE_SYNCBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device_id]
        )
    
    logger.info("Prepare Checkpoint")
    checkpointer = Checkpointer(
        cfg, model, optimizer, scheduler, output_dir
    )
    logger.info("Finish Checkpoint Preparation")

    logger.info("Prepare Datasets")
    data_loader = make_data_loader(
        cfg, is_train=True, is_distributed=num_gpus > 1,
        start_iter=checkpointer.iteration
    )
    data_loader_val = make_data_loader(
        cfg, is_train=False, is_distributed=num_gpus > 1,
        is_for_period=True
    )
    logger.info("Finish Datasets Preparation")
   
    arguments = dict(
        output_dir=output_dir,
        checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD, 
        test_period=cfg.SOLVER.TEST_PERIOD
    )

    do_train(model, data_loader, data_loader_val,
        optimizer, scheduler, checkpointer, arguments)

    return model

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
    
    output_dir = args.config_file.replace("configs", "outputs").strip('.yaml')
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
    
    output_config_path = os.path.join(output_dir, 'config.yaml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    model = train(cfg, args.device_id, args.num_gpus, output_dir, logger)
    
if __name__ == "__main__":
    main()
