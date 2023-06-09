#!/usr/bin/env python
# coding=utf-8

import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
import logging
import numpy as np
import libs.configs.default as cfg
from libs.data.layoutclmV30 import create_dataset
from libs.data.layoutclmV30.dataset import DataCollator

import transformers
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import PreTrainer as Trainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from layoutlmft.models.layoutclmV30 import LayoutCLMv30ForPretrain
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'pytorch_model.bin')):
        model = LayoutCLMv30ForPretrain.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = LayoutCLMv30ForPretrain(config)

    # Initialize our pretrain dataset and valid dataset
    train_npy_paths = []
    train_image_dirs = []
    valid_npy_paths = []
    valid_image_dirs = []
    for dataset in config.datasets:
        train_npy_paths.extend(cfg.datasets['train_%s_npy_paths' % dataset])
        train_image_dirs.extend(cfg.datasets['train_%s_image_dirs' % dataset])
        valid_npy_paths.extend(cfg.datasets['valid_%s_npy_paths' % dataset])
        valid_image_dirs.extend(cfg.datasets['valid_%s_image_dirs' % dataset])
    train_dataset = create_dataset(train_npy_paths, train_image_dirs, config)
    valid_dataset = create_dataset(valid_npy_paths, valid_image_dirs, config)
    data_collator = DataCollator()

    # Define our compute metrics function 
    def compute_metrics(p):
        predictions, labels = p
        preds_type = ["dtc_acc", "bdp_acc"]
        acc = dict()
        for pred_type, pred, label in zip(preds_type, predictions, labels):
            try:
                pred = pred.reshape(label.shape)
                mask = label != -100
                pred = np.logical_and(pred==label, mask)
                correct_nums = pred.sum()
                total_nums = max(mask.sum(), 1e-6)
                acc[pred_type] = correct_nums / total_nums
            except:
                acc[pred_type] = 0.0
        acc_avg = 0
        for v in acc.values():
            acc_avg += v
        acc_avg /= len(acc)
        acc['acc_avg'] = acc_avg
        return acc

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()