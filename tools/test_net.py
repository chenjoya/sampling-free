# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from sampling_free.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
from collections import OrderedDict
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from sampling_free.config import cfg
from sampling_free.data import make_data_loader
from sampling_free.engine.inference import inference
from sampling_free.modeling.detector import build_detection_model
from sampling_free.utils.checkpoint import DetectronCheckpointer
from sampling_free.utils.collect_env import collect_env_info
from sampling_free.utils.comm import synchronize, get_rank
from sampling_free.utils.logger import setup_logger
from sampling_free.utils.miscellaneous import mkdir
from sampling_free.utils.model_serialization import load_state_dict

def eval_dir(args, cfg, model, distributed, keep_best_only=False):
    last_eval_time = time.time()
    eval_file_list = []
    eval_done_file_list = []
    lasting_time = 6 * 3600  # 6 hours
    eval_interval = 5 * 60  # 5 minutes

    best_ap = 0
    not_best_model_path = None
    best_model_path = None

    ckpt_dir = args.eval_dir
    while True:
        start_time = time.time()
        eval_file_list = [
                fname for fname in os.listdir(ckpt_dir) if fname[-4:] == '.pth' \
                    and 'final' not in fname
        ]

        step_eval = {}
        for eval_file in eval_file_list:
            if eval_file in eval_done_file_list:
                continue

            try:
                step_pos_start = eval_file.find('model_') + 6
                step_pos_end = eval_file[step_pos_start:].find('.')
                step = eval_file[step_pos_start:step_pos_start + step_pos_end]
                step = int(step)
                #print("step: {}".format(step))
                step_eval[step] = eval_file
            except:
                print("failed to parse step for file {}.".format(eval_file))
                step = None
        step_eval = OrderedDict(sorted(step_eval.items()))

        for step, eval_file in step_eval.items():
            print("step: {}".format(step))
            model_path = os.path.join(ckpt_dir, eval_file)
            for n_try in range(3):
                try:
                    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                    load_state_dict(model, checkpoint.pop("model"))
                    ap = eval_checkpoint(cfg, model, ckpt_dir, distributed, n_iter=step)
                    eval_done_file_list.append(eval_file)
                    if best_ap < ap:
                        best_ap = ap
                        not_best_model_path = best_model_path
                        best_model_path = model_path
                    else:
                        not_best_model_path = model_path

                    if keep_best_only and not_best_model_path is not None:
                        os.remove(not_best_model_path)

                    print ('best ap: {}'.format(best_ap))
                    last_eval_time = time.time()
                    break
                except:
                    if n_try < 2:
                        print("failed to load/evaluate the checkpoint file {}."
                              " retrying after 10 seconds.".format(step))
                        time.sleep(10)
            else:
                print("failed to load/evaluate checkpoint files 3 times. skip this one.")

        cur_time = time.time()
        if cur_time - last_eval_time > lasting_time:
            print("Finishing testing dir since no more model files detected"
                  "in last 6 hours.")
            return

        time_to_next_eval = start_time + eval_interval - cur_time
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)

def eval_checkpoint(cfg, model, output_dir, distributed, n_iter=None):
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        box_only = (False if cfg.MODEL.PAA_ON or
                             cfg.MODEL.ATSS_ON or
                             cfg.MODEL.FCOS_ON or
                             cfg.MODEL.RETINANET_ON
                          else cfg.MODEL.RPN_ONLY)
        results = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=box_only,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        if results is not None:
            writer = SummaryWriter(output_dir)
            for ap_type, ap in results[0].results['bbox'].items():
                writer.add_scalar('{}_{}'.format(dataset_name, ap_type), float(ap), n_iter)
            writer.close()
            return float(results[0].results['bbox']['AP'])
        else:
            return 0


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--eval_dir', dest='eval_dir', help='evaluate all checkpoints in eval_dir', type=str)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR

    if args.eval_dir:
        eval_dir(args, cfg, model, distributed)
    else:
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        cname = checkpointer.get_checkpoint_file()
        try:
            n_iter = int(cname[cname.find('model_') + 6:cname.find('.pth')])
        except:
            n_iter = 0
        eval_checkpoint(cfg, model, output_dir, distributed, n_iter)

if __name__ == "__main__":
    main()
