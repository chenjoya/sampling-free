
import logging, os
from tqdm import tqdm

import torch

from sampling_free.config import cfg
from sampling_free.data.datasets.evaluation import evaluate
from sampling_free.utils import is_main_process, get_world_size, all_gather, synchronize, Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from .bbox_aug_vote import im_detect_bbox_aug_vote


def compute_on_dataset(model, data_loader, timer=None):
    model.eval()
    results_dict = {}
    device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                if cfg.TEST.BBOX_AUG.VOTE:
                    output = im_detect_bbox_aug_vote(model, images, device)
                else:
                    output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("sampling_free.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def do_inference(
        model,
        data_loader,
        iou_types=("bbox",),
        box_only=False,
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    num_gpus = get_world_size()
    logger = logging.getLogger("sampling_free.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(
        dataset.__class__.__name__, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_gpus / len(dataset), num_gpus
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_gpus / len(dataset),
            num_gpus,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        **extra_args)
