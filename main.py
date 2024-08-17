import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
from model.model_interface import MInterface
from data.data_interface import DInterface
import torch
from utils.utils import load_config, save_all_results_to_csv, set_num_classes
import datetime
import os


def run_loop_settings(args):
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random seed is reset to the initial cfg.seed value for each run iteration.
    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(args.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [args.seed + x for x in range(num_iterations)]
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(args.run_multiple_splits)
        seeds = [args.seed] * num_iterations
    return run_ids, seeds

def load_loggers(logger_dir, current_time_str, seed, run_id):
    loggers = [
        TensorBoardLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}'),
        CSVLogger(logger_dir, name=f'{current_time_str}_{seed}_{run_id}')
    ]
    return loggers

def load_callbacks(ckpt_dir, current_time_str, seed, run_id):
    # 构建存储模型检查点的目录路径
    ckpt_path = f"{ckpt_dir}/{current_time_str}_{seed}_{run_id}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    callbacks = [
        plc.EarlyStopping(
            monitor='train_loss',
            mode='min',
            patience=50,
            min_delta=0.001
        ),
        plc.ModelCheckpoint(
            monitor='train_loss',
            dirpath=ckpt_path,
            filename='best-{epoch:02d}-{val_wasserstein_distance:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
    ]
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.set_default_dtype(torch.float32)
    all_results = []
    for run_id, seed in zip(*run_loop_settings(args)):
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        loggers = load_loggers(args.logger_dir, current_time_str, seed, run_id)
        callbacks = load_callbacks(args.ckpt_dir, current_time_str, seed, run_id)
        data_module = DInterface(data_config=args.data)
        model_module = MInterface(model_config=args.model, optim_config=args.optim)
        
        initial_ckpt_path = args.train.pretrained_ckpt
        
        trainer = Trainer(
            callbacks=callbacks, 
            logger=loggers, 
            max_epochs=args.train.epochs, 
            min_epochs=args.train.tolerance, 
            accelerator=args.train.accelerator, 
            devices=args.train.gpus if args.train.gpus else None, 
            enable_checkpointing=True,
        )
        if initial_ckpt_path:
            trainer.fit(model_module, datamodule=data_module, ckpt_path=initial_ckpt_path)
        else:
            trainer.fit(model_module, datamodule=data_module)
        
        best_checkpoint_path = callbacks[1].best_model_path
        test_results = trainer.test(model=model_module, datamodule=data_module, ckpt_path=best_checkpoint_path)
        for result in test_results:
            result_entry = {
                "run_id": run_id,
                "seed": seed,
                "current_time": current_time_str,
                "best_checkpoint_path": best_checkpoint_path
            }
            result_entry.update(result)
            all_results.append(result_entry)
        
    result_file = f'{args.result_dir}/{args.data.name}/{args.data.dataset_size}.csv'
    save_all_results_to_csv(all_results, result_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    config = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    args = load_config(config.config)
    args = set_num_classes(args)
    main(args)