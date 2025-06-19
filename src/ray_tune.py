from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def ray_object_factory(func, default_cfg: DictConfig) -> Callable:
    """ Factory function to create a Ray object with a default configuration.
    """
    def object(cfg: dict):
        default_cfg.update(cfg)
        return func(default_cfg)
    return object

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks.append(RayTrainReportCallback())

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, plugins=[RayLightningEnvironment()], strategy=RayDDPStrategy())
    trainer = prepare_trainer(trainer)
    
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

def test_train(cfg: DictConfig):
    print("Testing train function...")
    log.info(f"Config: {cfg}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="ray_tune.yaml")
def main(cfg: DictConfig):
    """Main entry point for ray tune.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
#     search_space = {
#     "layer_1_size": tune.choice([32, 64, 128]),
#     "layer_2_size": tune.choice([64, 128, 256]),
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "batch_size": tune.choice([32, 64]),
# }
    ray_cfg = cfg.get("ray", None)
    if ray_cfg is None:
        log.error("Ray configuration not found! <cfg.ray=null>")
        return None

    search_space = ray_cfg.get("search_space", None)
    if search_space is None:
        log.error("Search space not found! <cfg.search_space=null>")
        return None
    log.info(f"Search space: {search_space}")
    # Safely parse the search space from config strings to avoid security risks with eval()
    allowed_tune_functions = {
        "choice": tune.choice,
        "randint": tune.randint,
        "uniform": tune.uniform,
        "quniform": tune.quniform,
        "loguniform": tune.loguniform,
        "qloguniform": tune.qloguniform,
    }
    import ast
    for key, value in search_space.items():
        if isinstance(value, str) and value.startswith("tune."):
            try:
                func_name_str = value.split(".")[1].split("(")[0]
                args_str = value[value.find("(") + 1 : value.rfind(")")]

                if func_name_str not in allowed_tune_functions:
                    raise ValueError(f"Unsupported tune function: {func_name_str}")

                # Safely evaluate arguments by wrapping them in a list
                args = ast.literal_eval(f"[{args_str}]")

                # Call the tune function with the parsed arguments
                search_space[key] = allowed_tune_functions[func_name_str](*args)
            except Exception as e:
                log.error(f"Failed to parse search space parameter '{key}: {value}'. Error: {e}")
                raise
    log.info("Search space parsed successfully.")
    
    scaling_config = hydra.utils.instantiate(ray_cfg.scaling_config)
    if scaling_config is None:
        log.error("Scaling configuration not found! <cfg.ray.scaling_config=null>")
        return None
    log.info(f"Scaling configuration: {scaling_config}")

    from ray.train.torch import TorchTrainer
    
    # trainable = ray_object_factory(train, cfg)
    trainable = ray_object_factory(test_train, cfg)
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        trainable,
        scaling_config=scaling_config,
        # run_config=run_config,
    )
    tune_config = hydra.utils.instantiate(ray_cfg.tune_config)
    if tune_config is None:
        log.error("Tune configuration not found! <cfg.ray.tune_config=null>")
        return None
    log.info(f"Tune configuration: {tune_config}")
    
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune_config,
    )
    results = tuner.fit()
    log.info("Ray Tune finished successfully!")
    best_result = results.get_best_result(
        metric=ray_cfg.tune_config.metric,
        mode=ray_cfg.tune_config.mode,
    )
    log.info(f"Best result: {best_result}")

    # train the model
    # metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )
    # return optimized metric
    # return metric_value
    return 


if __name__ == "__main__":
    results = main()
    results.get_best_result(metric="ptl/val_accuracy", mode="max")
