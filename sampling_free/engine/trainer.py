import datetime, logging, time

import torch

from .inference import do_inference
from sampling_free.utils import MetricLogger, reduce_dict

def do_train(
    model,
    data_loader,
    data_loaders_val,
    optimizer,
    scheduler,
    checkpointer,
    arguments
):
    logger = logging.getLogger("sampling-free.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iteration = len(data_loader)
    model.train()
    start_training_time, end = time.time(), time.time()

    for _iteration, batches in enumerate(data_loader, checkpointer.iteration):
        data_time = time.time() - end
        iteration = _iteration + 1

        optimizer.zero_grad()
        loss_dict = model(batches)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        scheduler.step()
        batch_time = time.time() - end

        if iteration % 100 == 0 or iteration == max_iteration:
            meters.update(time=batch_time, data=data_time)
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iteration=iteration,
                    max_iteration=max_iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        if iteration % arguments["checkpoint_period"] == 0:
            checkpointer.save(iteration)
        
        if iteration % arguments["test_period"] == 0:
            model.eval()
            do_inference(model, data_loaders_val, output_folder=arguments["output_dir"])
            model.train()
        
        torch.cuda.synchronize()
        end = time.time()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / max_iteration
        )
    )
