# coding=utf-8
import torch
import copy
from torch import nn
from typing import Optional
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.functional import embedding
from transformers.models.auto.configuration_auto import AutoConfig
from libs.model.extractor import RoiFeatExtraxtor
from libs.configs.default import counter
import itertools
import detectron2
from .swin_transformer import VisionBackbone
import os
import json
import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from ..layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Layer
from ..layoutlmv2.modeling_layoutlmv2 import *
from transformers.models.layoutlm.modeling_layoutlm import LayoutLMPooler as GraphDocPooler
from .configuration_graphdoc import GraphDocConfig
from .utils_docile import extract_item_feats, parse_item_merge_labels, extract_merge_feats_v2, parse_merge_labels, select_items_entitys_idx, decode_merge_logits, cal_tp_total
import libs.configs.doclie_config_ocr_data_lir_merge as cfg
from .modeling_graphdoc import GraphDocForTokenClassification, AttentionMerger

class LirMergeTrain(GraphDocForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.merge_head_item = AttentionMerger(config.hidden_size, config.hidden_size)
        self.vocab = cfg.Doclielir_vocab


    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        instance_label_batch=None,
        instance_class_batch=None,
        line_item_label_batch=None,
        image_infos=None,
        output_path=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.layoutclm(
            input_sentences=input_sentences,
            input_sentences_masks=input_sentences_masks,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
            
        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output) # (batch_size, num_boxes, c)
        classify_logits = self.classifier(sequence_output)
        output_logits = classify_logits
        
        if self.training: 
                
            instance_entity_features, instance_merge_mask = extract_merge_feats_v2(sequence_output, instance_label_batch, classify_logits)
            instance_merger_logits = self.merge_head(instance_entity_features, instance_merge_mask)
            instance_merge_labels, instance_merge_label_mask = parse_merge_labels(sequence_output, instance_label_batch)  #制作label（B, ocr_box.num in img_instances, ocr_box.num in img_instances）
            instance_raw_merge_loss = F.binary_cross_entropy_with_logits(instance_merger_logits, instance_merge_labels, reduction='none')
            instance_merge_loss = (instance_raw_merge_loss * instance_merge_label_mask).sum() / instance_merge_label_mask.sum()

            item_entity_features, item_merge_mask = extract_item_feats(sequence_output, line_item_label_batch, classify_logits)
            item_merger_logits = self.merge_head_item(item_entity_features, item_merge_mask)
            item_merge_labels, item_merge_label_mask = parse_item_merge_labels(sequence_output, line_item_label_batch)  #制作label（B, ocr_box.num in img_instances, ocr_box.num in img_instances）
            item_raw_merge_loss = F.binary_cross_entropy_with_logits(item_merger_logits, item_merge_labels, reduction='none')
            item_merge_loss = (item_raw_merge_loss * item_merge_label_mask).sum() / item_merge_label_mask.sum()

            active_loss = attention_mask.view(-1) == 1
            active_logits = classify_logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1, self.num_labels)[active_loss]
            active_labels = active_labels.type_as(active_logits)
            classify_loss = self.bce_logits(active_logits, active_labels)

            loss = classify_loss + instance_merge_loss + item_merge_loss

            if not return_dict:
                output = (output_logits.sigmoid(), ) + outputs[2:] 
                return ((loss, classify_loss, instance_merge_loss, item_merge_loss) + output) if loss is not None else output



class LirMergeTrainEachClass(GraphDocForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.merge_head_item = AttentionMerger(config.hidden_size, config.hidden_size)
        self.vocab = cfg.Doclielir_vocab


    def forward(
        self,
        input_sentences=None,
        input_sentences_masks=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        instance_label_batch=None,
        instance_class_batch=None,
        line_item_label_batch=None,
        image_infos=None,
        output_path=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.layoutclm(
            input_sentences=input_sentences,
            input_sentences_masks=input_sentences_masks,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
            
        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output) # (batch_size, num_boxes, c)
        classify_logits = self.classifier(sequence_output)
        output_logits = classify_logits
        
        if self.training: 
            instance_merge_loss = 0
            sum_instance_merge_loss = 0
            sum_instance_merge_mask = 0
            have_instance = False
            for class_idx, class_name in enumerate(self.vocab.key_words):
                specific_class_instance_label = []
                specific_class_instance_class = []
                for i, (instance_labels, instance_classes) in enumerate(zip(instance_label_batch, instance_class_batch)):
                    instance_labels_tmp = []
                    instance_classes_tmp = []
                    for instance_label, instance_class in zip(instance_labels, instance_classes):
                        if instance_class == class_name:
                            instance_labels_tmp.append(instance_label)
                            instance_classes_tmp.append(instance_class)
                    specific_class_instance_label.append(instance_labels_tmp)
                    specific_class_instance_class.append(instance_classes_tmp)
                if sum([len(batch_li) for batch_li in specific_class_instance_class]) != 0:
                    have_instance = True
                    instance_entity_features, instance_merge_mask = extract_merge_feats_v2(sequence_output, specific_class_instance_label, classify_logits)
                    instance_merger_logits = self.merge_head(instance_entity_features, instance_merge_mask)
                    instance_merge_labels, instance_merge_label_mask = parse_merge_labels(sequence_output, specific_class_instance_label)  
                    instance_raw_merge_loss = F.binary_cross_entropy_with_logits(instance_merger_logits, instance_merge_labels, reduction='none')
                    sum_instance_merge_loss += (instance_raw_merge_loss * instance_merge_label_mask).sum() 
                    sum_instance_merge_mask += instance_merge_label_mask.sum()
            
            if have_instance:
                instance_merge_loss = sum_instance_merge_loss / sum_instance_merge_mask
            else:
                instance_merge_loss = torch.tensor(0.0, requires_grad=True, device=classify_logits.device)
                
            item_entity_features, item_merge_mask = extract_item_feats(sequence_output, line_item_label_batch, classify_logits)
            item_merger_logits = self.merge_head_item(item_entity_features, item_merge_mask)
            item_merge_labels, item_merge_label_mask = parse_item_merge_labels(sequence_output, line_item_label_batch)  
            item_raw_merge_loss = F.binary_cross_entropy_with_logits(item_merger_logits, item_merge_labels, reduction='none')
            item_merge_loss = (item_raw_merge_loss * item_merge_label_mask).sum() / item_merge_label_mask.sum()

            active_loss = attention_mask.view(-1) == 1
            active_logits = classify_logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1, self.num_labels)[active_loss]
            active_labels = active_labels.type_as(active_logits)
            classify_loss = self.bce_logits(active_logits, active_labels)

            loss = classify_loss + instance_merge_loss + item_merge_loss

            if not return_dict:
                output = (output_logits.sigmoid(), ) + outputs[2:] 
                return ((loss, classify_loss, instance_merge_loss, item_merge_loss) + output) if loss is not None else output

            