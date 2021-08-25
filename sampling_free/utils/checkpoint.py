import logging, os

import torch

from .model_serialization import load_state_dict
from .c2_model_loading import load_c2_format
from .imports import import_file
from .model_zoo import cache_url

class Checkpointer(object):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.paths_catalog = cfg.PATHS_CATALOG
        self.iteration = 0
        self.logger = logging.getLogger(__name__)

        p, f = cfg.MODEL.PRETRAINED, cfg.MODEL.WEIGHT
        if not f:
            f = self.get_checkpoint_file() if self.has_checkpoint() else p
        if not f and not p:
            self.logger.info("No checkpoint found. Ignore checkpoint initialization.")
            return
        
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(cfg, f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        if "iteration" in checkpoint:
            self.iteration = checkpoint["iteration"]
       
    def save(self, iteration):
        assert self.save_dir 
        data = dict(model=self.model.state_dict())
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data["iteration"] = iteration

        save_file = os.path.join(self.save_dir, "{}.pth".format("model_{:07d}".format(iteration)))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)
    
    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, cfg, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "sampling_free.config.paths_catalog", self.paths_catalog, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(cfg, f)
        loaded = torch.load(f, map_location=torch.device("cpu"))
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))

