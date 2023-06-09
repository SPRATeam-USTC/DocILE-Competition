#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import torch
import pandas as pd
import numpy as np
sys.path.append('./')

import libs.configs.doclie_config_ocr_data_lir_merge as cfg
from libs.data.doclie_lir_merge_old_box import create_dataset, DataCollator

import layoutlmft
import transformers
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import DocileLirTrainer as Trainer
from transformers import (
    AutoModel,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from layoutlmft.models.graphdoc.modeling_lir_train_val import LirMergeTrain



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
    logger.setLevel(logging.INFO)
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
        num_labels=len(cfg.Doclielir_vocab),
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.sentence_model)
    sentence_bert = AutoModel.from_pretrained(config.sentence_model)
    model = LirMergeTrain.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # init train / eval dataset
    train_dataset = create_dataset(cfg.train_npy_paths, cfg.Doclielir_vocab, cfg)
    eval_dataset = create_dataset(cfg.valid_npy_paths, cfg.Doclielir_vocab, cfg)

    # Data collator
    data_collator = DataCollator()

    # Metrics
    class compute_metrics:
        def __init__(self, vocab):
            self.vocab = vocab
            self.use_type = 'merger'
            # self.use_type = 'classify'
        
        def __call__(self, p):
            if self.use_type == 'merger':
                (metrics_origin_tmp, predictions_origin), labels_origin = p
                threshold = 0.5
                labels = []
                predictions = []
                labels_tmp = np.reshape(labels_origin, (-1, labels_origin.shape[2]))
                predictions_tmp = np.reshape(predictions_origin, (-1, predictions_origin.shape[2]))
                
                for i,item in enumerate(labels_tmp):
                    if sum(item) > 0:
                        labels.append(item)
                        predictions.append(predictions_tmp[i])

                labels = np.array(labels)
                predictions = np.array(predictions)
                predictions[predictions > threshold] = 1
                predictions[predictions <= threshold] = 0

                tp_tmp = np.sum(predictions * labels, axis=0)
                fp_tmp = np.sum(predictions * (1-labels), axis=0)
                fn_tmp = np.sum((1-predictions) * labels, axis=0)
                tp_classify = np.delete(tp_tmp, [7,18,22])
                fp_classify = np.delete(fp_tmp, [7,18,22])
                fn_classify = np.delete(fn_tmp, [7,18,22])
                
                precision_classify = np.nan_to_num(tp_classify / (tp_classify + fp_classify))
                recall_classify = np.nan_to_num(tp_classify / (tp_classify + fn_classify))
                f1_classify = np.nan_to_num(2 * precision_classify * recall_classify / (precision_classify + recall_classify))
                macro_f1 = np.mean(f1_classify)
                TP = np.sum(tp_classify)
                FP = np.sum(fp_classify)
                FN = np.sum(fn_classify)
                micro_precision = np.nan_to_num(TP / (TP + FP))
                micro_recall = np.nan_to_num(TP / (TP + FN))
                micro_F1 = np.nan_to_num(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))
                data_classify = {
                        'class name': [i for j, i in enumerate(self.vocab.key_words) if j not in [7,18,22]],
                        'precision': [round(x, 4) for x in precision_classify],
                        'recall': [round(x, 4) for x in recall_classify],
                        'f1': [round(x, 4) for x in f1_classify],
                        'tp': [int(x) for x in tp_classify],
                        'fp': [int(x) for x in fp_classify],
                        'fn': [int(x) for x in fn_classify],
                    }
                df_classify = pd.DataFrame(data_classify)
                df_classify.sort_values("f1",inplace=True)
                
                table_classify = df_classify.to_string(index=False)
                results_classify = {
                    "macro_f1": macro_f1,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall,
                    "micro_F1": micro_F1,
                }
                
                metrics_origin = np.delete(metrics_origin_tmp, [7,18,22], axis=1)
                metrics_single_sample = metrics_origin.sum(axis=1)
                tp, pred_num, gt_num = metrics_single_sample.sum(axis=0)
                P = tp / pred_num if pred_num > 0 else -1
                R = tp / gt_num if gt_num > 0 else -1
                F1 = 2 * P * R / (P + R) if P + R > 0 else -1

                metrics_single_class = metrics_origin.sum(axis=0)
                tp_single_class = metrics_single_class[:, 0]
                pred_num_single_class = metrics_single_class[:, 1]
                gt_num_single_class = metrics_single_class[:, 2]
                precision = np.nan_to_num(tp_single_class / pred_num_single_class)
                recall = np.nan_to_num(tp_single_class / gt_num_single_class)
                f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
                data = {
                        'class name': [i for j, i in enumerate(self.vocab.key_words) if j not in [7,18,22]],
                        'precision': [round(x, 4) for x in precision],
                        'recall': [round(x, 4) for x in recall],
                        'f1': [round(x, 4) for x in f1],
                        'tp_num': [int(x) for x in tp_single_class],
                        'pred_num': [int(x) for x in pred_num_single_class],
                        'gt_num': [int(x) for x in gt_num_single_class],
                    }
                df = pd.DataFrame(data)
                df.sort_values("f1",inplace=True)

                table = df.to_string(index=False)
                results = {
                    "F1": F1,
                    "P": P,
                    "R": R 
                }

                logger.info("***** Eval results *****")
                logger.info("***** Eval classify results *****")
                logger.info("\n"+table_classify)
                for key in sorted(results_classify.keys()):
                    logger.info("  %s = %s", key, str(results_classify[key]))
                logger.info("***** Eval merge results *****")
                
                logger.info("\n"+table)
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))

                return results
            else:
                predictions_origin, labels_origin = p
                threshold = 0.5
                labels = []
                predictions = []
                labels_tmp = np.reshape(labels_origin, (-1, labels_origin.shape[2]))
                predictions_tmp = np.reshape(predictions_origin, (-1, predictions_origin.shape[2]))
                
                for i,item in enumerate(labels_tmp):
                    if sum(item) > 0:
                        labels.append(item)
                        predictions.append(predictions_tmp[i])

                labels = np.array(labels)
                predictions = np.array(predictions)
                predictions[predictions > threshold] = 1
                predictions[predictions <= threshold] = 0

                tp_tmp = np.sum(predictions * labels, axis=0)
                fp_tmp = np.sum(predictions * (1-labels), axis=0)
                fn_tmp = np.sum((1-predictions) * labels, axis=0)
                tp = np.delete(tp_tmp, [7,18,22])
                fp = np.delete(fp_tmp, [7,18,22])
                fn = np.delete(fn_tmp, [7,18,22])
                
                precision = np.nan_to_num(tp / (tp + fp))
                recall = np.nan_to_num(tp / (tp + fn))
                f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
                macro_f1 = np.mean(f1)
                TP = np.sum(tp)
                FP = np.sum(fp)
                FN = np.sum(fn)
                micro_precision = np.nan_to_num(TP / (TP + FP))
                micro_recall = np.nan_to_num(TP / (TP + FN))
                micro_F1 = np.nan_to_num(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall))
                data = {
                        'class name': [i for j, i in enumerate(self.vocab.key_words) if j not in [7,18,22]],
                        'precision': [round(x, 4) for x in precision],
                        'recall': [round(x, 4) for x in recall],
                        'f1': [round(x, 4) for x in f1],
                        'tp': [int(x) for x in tp],
                        'fp': [int(x) for x in fp],
                        'fn': [int(x) for x in fn],
                    }
                df = pd.DataFrame(data)
                df.sort_values("f1",inplace=True)
                table = df.to_string(index=False)
                results = {
                    "macro_f1": macro_f1,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall,
                    "micro_F1": micro_F1,
                }
                logger.info("***** Eval results *****")
                logger.info("\n"+table)
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))
                return results
          
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(cfg.Doclielir_vocab),
    )

   


    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

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
