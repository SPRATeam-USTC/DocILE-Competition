# coding=utf-8
import torch
import copy
from torch import nn
from typing import Optional
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.functional import embedding
from transformers.models.auto.configuration_auto import AutoConfig
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
from .utils_docile import select_instances_polys_idx_fusion, select_items_entitys_idx_fusion, select_instances_polys_idx, extract_item_feats,extract_merge_feats_v2, select_items_entitys_idx, decode_merge_logits
import libs.configs.classify_threshold_lir as cfg_lir
import libs.configs.classify_threshold_kie as cfg_kie

from .modeling_graphdoc import GraphDocForTokenClassification
from .modeling_lir_train_val import LirMergeTrain

class InferKieMerge(GraphDocForTokenClassification):
    pass

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
        image_infos=None,
        output_path=None
    ):

        assert image.shape[0] == 1
        
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
        ) # graphdoc提特征

        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output) # (batch_size, num_boxes, c)
        classify_logits = self.classifier(sequence_output)
        output_logits = classify_logits
        
        if not self.training: 
            pred_items_class_img = []
            pred_items_polys_img = []
            pred_items_scores_img = []
            for class_idx, class_name in enumerate(self.vocab.key_words):
                valid_items_polys_idxes = select_items_entitys_idx(class_idx, classify_logits, attention_mask, cfg_kie.classify_threshold)  # 对ocr_box,先看他属于哪个类别，把class in ["NAME", "CNT", "PRICE", "PRICE&CNT", "CNT&NAME"]的挑出来
                entity_features, merge_mask = extract_merge_feats_v2(sequence_output, valid_items_polys_idxes, classify_logits)
                if sum([len(batch_li) for batch_li in merge_mask]) != 0:
                    merger_logits = self.merge_head(entity_features, merge_mask)
                    pred_items_polys_idxes, pred_items_polys_scores = decode_merge_logits(merger_logits, valid_items_polys_idxes, classify_logits, self.vocab)
                    pred_items_polys_img.extend(pred_items_polys_idxes[0])
                    pred_items_scores_img.extend(pred_items_polys_scores[0])
                    pred_items_class_img.extend([class_name for _ in range(len(pred_items_polys_idxes[0]))])

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            classify_id_preds = (classify_logits[0][1:].sigmoid() > cfg_kie.classify_threshold).int()
            classify_preds = []
            for x in classify_id_preds:
                if x.sum() > 0:
                    classify_preds.append(self.vocab.ids_to_words(torch.where(x==1)[0].cpu().tolist()))
                else:
                    classify_preds.append(["other"])
            merger_preds_instances = pred_items_polys_img
            merger_preds_classes = pred_items_class_img
            merger_preds_scores = [float(x) for x in pred_items_scores_img]
            ouput_dict = {
                'classify_preds': classify_preds,
                'classify_logits': classify_logits[0][1:].sigmoid().cpu().tolist(),
                'merger_preds_instances': merger_preds_instances,
                'merger_preds_classes': merger_preds_classes,
                'merger_preds_scores': merger_preds_scores,
                'gt_info':image_infos,
                
            }

            with open(os.path.join(output_path, os.path.basename(image_infos['image_name'])[:-4] + '.json'), 'w') as f:
                json.dump(ouput_dict, f, ensure_ascii=False, indent=2)





class InferKieMergeFusion(GraphDocForTokenClassification):
    pass

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
        image_infos=None,
        output_path=None,
        logit=None,
        classify_pred=None
    ):

        assert image.shape[0] == 1
        
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
        ) # graphdoc提特征

        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output) # (batch_size, num_boxes, c)
        classify_logits = logit
        classify_preds = classify_pred
        
        if not self.training: 
            pred_items_class_img = []
            pred_items_polys_img = []
            pred_items_scores_img = []
            for class_idx, class_name in enumerate(self.vocab.key_words):
                valid_items_polys_idxes = select_items_entitys_idx_fusion(class_name, classify_pred)  # 对ocr_box,先看他属于哪个类别，把class in ["NAME", "CNT", "PRICE", "PRICE&CNT", "CNT&NAME"]的挑出来
                entity_features, merge_mask = extract_merge_feats_v2(sequence_output, valid_items_polys_idxes, classify_logits)
                if sum([len(batch_li) for batch_li in merge_mask]) != 0:
                    merger_logits = self.merge_head(entity_features, merge_mask)
                    pred_items_polys_idxes, pred_items_polys_scores = decode_merge_logits(merger_logits, valid_items_polys_idxes, classify_logits, self.vocab)
                    pred_items_polys_img.extend(pred_items_polys_idxes[0])
                    pred_items_scores_img.extend(pred_items_polys_scores[0])
                    pred_items_class_img.extend([class_name for _ in range(len(pred_items_polys_idxes[0]))])

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            merger_preds_instances = pred_items_polys_img
            merger_preds_classes = pred_items_class_img
            merger_preds_scores = [float(x) for x in pred_items_scores_img]
            ouput_dict = {
                'classify_preds': classify_preds,
                'classify_logits': classify_logits.tolist(),
                'merger_preds_instances': merger_preds_instances,
                'merger_preds_classes': merger_preds_classes,
                'merger_preds_scores': merger_preds_scores,
                'gt_info':image_infos,
                
            }

            with open(os.path.join(output_path, os.path.basename(image_infos['image_name'])[:-4] + '.json'), 'w') as f:
                json.dump(ouput_dict, f, ensure_ascii=False, indent=2)



class InferLirMerge(LirMergeTrain):
    pass

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_infos=None,
        output_path=None
    ):

        assert image.shape[0] == 1
        
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
        ) # graphdoc提特征

        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output) # (batch_size, num_boxes, c)
        classify_logits = self.classifier(sequence_output)
        output_logits = classify_logits

        if not self.training: 
            instances_polys_idxes = select_instances_polys_idx(classify_logits, attention_mask, cfg_lir.classify_threshold) 
            if not all(not x for sublst in instances_polys_idxes for subsublst in sublst for x in subsublst):
                instances_polys_features, instances_merge_mask = extract_merge_feats_v2(sequence_output, instances_polys_idxes, classify_logits)
                instance_merger_logits = self.merge_head(instances_polys_features, instances_merge_mask)
                pred_instances_polys_idxes, pred_instances_polys_scores = decode_merge_logits(instance_merger_logits, instances_polys_idxes, classify_logits, self.vocab)
                
                if not all(not x for sublst in pred_instances_polys_idxes for subsublst in sublst for x in subsublst):
                    item_entity_features, item_merge_mask = extract_item_feats(sequence_output, [pred_instances_polys_idxes], classify_logits)
                    item_merger_logits = self.merge_head_item(item_entity_features, item_merge_mask)
                    item_instances_idxes = [[[x for pred_instances_polys_idxes_bi in pred_instances_polys_idxes for x in range(len(pred_instances_polys_idxes_bi))]]]
                    pred_item_instances_idxes, pred_item_instances_scores = decode_merge_logits(item_merger_logits, item_instances_idxes, classify_logits, self.vocab)


                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    classify_id_preds = (classify_logits[0][1:].sigmoid() > cfg_lir.classify_threshold).int()
                    classify_preds = []
                    for x in classify_id_preds:
                        if x.sum() > 0:
                            classify_preds.append(self.vocab.ids_to_words(torch.where(x==1)[0].cpu().tolist()))
                        else:
                            classify_preds.append(["other"])
                    pred_instances_polys_idxes = pred_instances_polys_idxes[0]
                    pred_instances_polys_scores = [float(x) for x in pred_instances_polys_scores[0]]
                    pred_item_instances_idxes = pred_item_instances_idxes[0]
                    pred_item_instances_scores = [float(x) for x in pred_item_instances_scores[0]]
                    ouput_dict = {
                        'classify_preds': classify_preds,
                        'classify_logits': classify_logits[0][1:].sigmoid().cpu().tolist(),
                        'pred_instances_polys_idxes': pred_instances_polys_idxes,
                        'pred_instances_polys_scores': pred_instances_polys_scores,
                        'pred_item_instances_idxes': pred_item_instances_idxes,
                        'pred_item_instances_scores': pred_item_instances_scores,
                        'gt_info':image_infos,
                        
                    }

                    with open(os.path.join(output_path, os.path.basename(image_infos['image_name'])[:-4] + '.json'), 'w') as f:
                        json.dump(ouput_dict, f, ensure_ascii=False, indent=2)






class InferLirMergeFusion(LirMergeTrain):
    pass

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_infos=None,
        output_path=None,
        logit=None,
        classify_pred=None
    ):

        assert image.shape[0] == 1
        
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
        ) # graphdoc提特征

        seq_length = inputs_embeds.size(1)
        sequence_output, image_output = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        sequence_output = self.dropout(sequence_output) # (batch_size, num_boxes, c)
        classify_logits = logit
        classify_preds = classify_pred

        if not self.training: 
            instances_polys_idxes = select_instances_polys_idx_fusion(classify_preds) 
            if not all(not x for sublst in instances_polys_idxes for subsublst in sublst for x in subsublst):
                instances_polys_features, instances_merge_mask = extract_merge_feats_v2(sequence_output, instances_polys_idxes, classify_logits)
                instance_merger_logits = self.merge_head(instances_polys_features, instances_merge_mask)
                pred_instances_polys_idxes, pred_instances_polys_scores = decode_merge_logits(instance_merger_logits, instances_polys_idxes, classify_logits, self.vocab)
                
                if not all(not x for sublst in pred_instances_polys_idxes for subsublst in sublst for x in subsublst):
                    item_entity_features, item_merge_mask = extract_item_feats(sequence_output, [pred_instances_polys_idxes], classify_logits)
                    item_merger_logits = self.merge_head_item(item_entity_features, item_merge_mask)
                    item_instances_idxes = [[[x for pred_instances_polys_idxes_bi in pred_instances_polys_idxes for x in range(len(pred_instances_polys_idxes_bi))]]]
                    pred_item_instances_idxes, pred_item_instances_scores = decode_merge_logits(item_merger_logits, item_instances_idxes, classify_logits, self.vocab)


                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    pred_instances_polys_idxes = pred_instances_polys_idxes[0]
                    pred_instances_polys_scores = [float(x) for x in pred_instances_polys_scores[0]]
                    pred_item_instances_idxes = pred_item_instances_idxes[0]
                    pred_item_instances_scores = [float(x) for x in pred_item_instances_scores[0]]
                    ouput_dict = {
                        'classify_preds': classify_preds,
                        'classify_logits': classify_logits.tolist(),
                        'pred_instances_polys_idxes': pred_instances_polys_idxes,
                        'pred_instances_polys_scores': pred_instances_polys_scores,
                        'pred_item_instances_idxes': pred_item_instances_idxes,
                        'pred_item_instances_scores': pred_item_instances_scores,
                        'gt_info':image_infos,
                        
                    }

                    with open(os.path.join(output_path, os.path.basename(image_infos['image_name'])[:-4] + '.json'), 'w') as f:
                        json.dump(ouput_dict, f, ensure_ascii=False, indent=2)

    