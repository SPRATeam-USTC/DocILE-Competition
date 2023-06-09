import torch
import numpy as np
from torch.nn import functional as F
import itertools
from operator import itemgetter

def align_logits(logits):
    batch_size = len(logits)
    max_length = max([_.shape[0] for _ in logits])
    dim = logits[0].shape[1]

    aligned_logits = torch.full((batch_size, max_length, dim), -100, dtype=logits[0].dtype, device=logits[0].device)
    for batch_idx, logits_pb in enumerate(logits):
        aligned_logits[batch_idx, :logits_pb.shape[0]] = logits_pb

    return aligned_logits




def extract_merge_feats_v2(bbox_features, items_polys_idxes, classify_logits):
    l_lst = [sum([len(t) for t in items_polys_idxes_bi]) for items_polys_idxes_bi in items_polys_idxes] 
    l_max = max(l_lst)
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    vocab_len = classify_logits.shape[-1]
    entity_features = torch.zeros((B, C, l_max), dtype=dtype, device=device)
    items_polys_idxes_batch = [list(itertools.chain(*items_polys_idxes_bi)) for items_polys_idxes_bi in items_polys_idxes] 
    for b_i in range(B):
        entity_index = torch.tensor(items_polys_idxes_batch[b_i], dtype=torch.long, device=device)
        temp_f = bbox_features[b_i, entity_index + 1]  # entity_index + 1: to remove 1st global image
        entity_features[b_i, :C, :len(entity_index)] = temp_f.permute(1, 0)
    merge_mask = torch.zeros((B, l_max), dtype=dtype, device=device)
    for b_i in range(B):
        merge_mask[b_i, :l_lst[b_i]] = 1
    return entity_features, merge_mask




def extract_item_feats(bbox_features, line_item_label_batch, classify_logits):
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    l_lst = [sum([len(t) for t in line_item_label]) for line_item_label in line_item_label_batch]  
    l_max = max(l_lst)
    entity_features = torch.zeros((B, C, l_max), dtype=dtype, device=device)
    for b_i,line_item_label in enumerate(line_item_label_batch):
        count = 0
        for j,item in enumerate(line_item_label):
            for instance in item:
                entity_index = torch.tensor(instance, dtype=torch.long, device=device)
                temp_f = bbox_features[b_i, entity_index + 1].mean(dim=0)
                entity_features[b_i, :C, count] = temp_f
                count += 1
    merge_mask = torch.zeros((B, l_max), dtype=dtype, device=device)
    for b_i in range(B):
        merge_mask[b_i, :l_lst[b_i]] = 1
    return entity_features, merge_mask

def extract_merge_feats(bbox_features, items_polys_idxes, classify_logits=None):
    l_lst = [sum([len(t) for t in items_polys_idxes_bi]) for items_polys_idxes_bi in items_polys_idxes]
    l_max = max(l_lst)
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    entity_features = torch.zeros((B, C, l_max), dtype=dtype, device=device)
    items_polys_idxes_batch = [list(itertools.chain(*items_polys_idxes_bi)) for items_polys_idxes_bi in items_polys_idxes]
    for b_i in range(B):
        entity_index = torch.tensor(items_polys_idxes_batch[b_i], dtype=torch.long, device=device)
        temp_f = bbox_features[b_i, entity_index + 1]  # entity_index + 1: to remove 1st global image
        entity_features[b_i, :C, :len(entity_index)] = temp_f.permute(1, 0)
        
    merge_mask = torch.zeros((B, l_max), dtype=dtype, device=device)
    for b_i in range(B):
        merge_mask[b_i, :l_lst[b_i]] = 1
    return entity_features, merge_mask

def extract_merge_feats_kie(bbox_features, instances_boxes_idxes_batch):
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    boxes_lst_batch = [list(set(sorted([num for row in instances_boxes_idxes for num in row]))) for instances_boxes_idxes in instances_boxes_idxes_batch]
    l_max = max([len(boxes_lst) for boxes_lst in boxes_lst_batch])

    entity_features = torch.zeros((B, C, l_max), dtype=dtype, device=device)
    for b_i in range(B):
        entity_index = torch.tensor(boxes_lst_batch[b_i], dtype=torch.long, device=device)
        temp_f = bbox_features[b_i, entity_index + 1]  # entity_index + 1: to remove 1st global image
        entity_features[b_i, :C, :len(entity_index)] = temp_f.permute(1, 0)
        
    return entity_features
    # return entity_features, merge_mask


def parse_merge_labels(bbox_features, items_polys_idxes):
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    l_lst = [sum([len(t) for t in items_polys_idxes_bi]) for items_polys_idxes_bi in items_polys_idxes]
    l_max = max(l_lst)
    merge_labels = torch.zeros((B, l_max, l_max), dtype=dtype, device=device) - 1
    for b_i in range(B):
        items_polys_idxes_bi = items_polys_idxes[b_i]
        items_len_lst = [len(t) for t in items_polys_idxes_bi]
        for items_i, items in enumerate(items_polys_idxes_bi):
            items_label = torch.zeros((l_max), dtype=dtype, device=device)
            items_label[sum(items_len_lst[:items_i]):sum(items_len_lst[:items_i + 1])] = 1
            merge_labels[b_i, :, sum(items_len_lst[:items_i]):sum(items_len_lst[:items_i + 1])] = items_label[:, None]
    merge_label_mask = torch.zeros((B, l_max, l_max), dtype=dtype, device=device)
    for b_i, l in enumerate(l_lst):
        merge_label_mask[b_i, :l, :l] = 1
    return merge_labels, merge_label_mask


def parse_item_merge_labels(bbox_features, line_item_label_batch):
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    l_lst = [sum([len(t) for t in line_item_label]) for line_item_label in line_item_label_batch]  
    l_max = max(l_lst)
    merge_labels = torch.zeros((B, l_max, l_max), dtype=dtype, device=device) - 1
    for b_i,line_item_label in enumerate(line_item_label_batch):
        items_len_lst = [len(t) for t in line_item_label]
        for items_i, items in enumerate(line_item_label):
            items_label = torch.zeros((l_max), dtype=dtype, device=device)
            items_label[sum(items_len_lst[:items_i]):sum(items_len_lst[:items_i + 1])] = 1
            merge_labels[b_i, :, sum(items_len_lst[:items_i]):sum(items_len_lst[:items_i + 1])] = items_label[:, None]
    merge_label_mask = torch.zeros((B, l_max, l_max), dtype=dtype, device=device)
    for b_i, l in enumerate(l_lst):
        merge_label_mask[b_i, :l, :l] = 1
    return merge_labels, merge_label_mask


def parse_merge_label_kie(bbox_features, instances_boxes_idxes_batch, instance_class_batch, key_words):
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    boxes_lst_batch = [sorted(list(set([num for row in instances_boxes_idxes for num in row]))) for instances_boxes_idxes in instances_boxes_idxes_batch]
    l_max = max([len(boxes_lst) for boxes_lst in boxes_lst_batch])
    merge_labels = torch.zeros((B, l_max, l_max), dtype=dtype, device=device) - 1
    merge_label_classes_mask_tmp = torch.zeros((B, len(key_words),l_max, l_max), dtype=dtype, device=device)
    merge_label_classes_mask = torch.zeros((B, len(key_words),l_max, l_max), dtype=dtype, device=device)

    for b_i in range(B):
        boxes_lst = boxes_lst_batch[b_i]
        merge_labels[b_i, :len(boxes_lst), :len(boxes_lst)] = 0
        # merge_label_mask[b_i, :len(boxes_lst), :len(boxes_lst)] = 1
        for instance_boxes_idxes, instance_class in zip(instances_boxes_idxes_batch[b_i], instance_class_batch[b_i]):
            class_idx = key_words.index(instance_class)
            index = torch.tensor([i for i,x in enumerate(boxes_lst) if x in instance_boxes_idxes])
            idx = torch.cat((index.repeat(len(index)), torch.repeat_interleave(index, len(index))))
            merge_labels[b_i, idx, idx.view(-1, 1)] = 1
            merge_label_classes_mask_tmp[b_i, class_idx, idx, idx.view(-1, 1)] = 1

    for b_i in range(B):
        for class_idx in range(len(key_words)):
            row_col = torch.nonzero(merge_label_classes_mask_tmp[b_i][class_idx])
            unique_elements = torch.unique(row_col.flatten())
            idx = torch.cat((unique_elements.repeat(len(unique_elements)), torch.repeat_interleave(unique_elements, len(unique_elements))))
            merge_label_classes_mask[b_i, class_idx, idx, idx.view(-1, 1)] = 1
            
    return merge_labels, merge_label_classes_mask

    # return merge_labels, merge_label_mask, boxes_lst_batch, merge_label_classes_mask

def select_items_entitys_idx(class_idx, classify_logits, attention_mask, threshold):
    B = classify_logits.shape[0]
    batch_select_idxes = [[] for _ in range(B)]
    for b_i in range(B):
        logit = classify_logits[b_i][attention_mask[b_i].bool()]
        logit = logit.sigmoid()
        pred_class_lst = torch.where(logit > threshold, 1, 0)
        indices = torch.where(pred_class_lst[:,class_idx]==1)[0].tolist()
        batch_select_idxes[b_i].append(indices)
    return batch_select_idxes

def select_items_entitys_idx_fusion(class_name, classify_pred) :
    B = 1
    batch_select_idxes = [[] for _ in range(B)]
    for b_i in range(B):
        indices = [id for id,classify_pred_i in enumerate(classify_pred) if class_name in classify_pred_i]
        batch_select_idxes[b_i].append(indices)
    return batch_select_idxes

def select_instances_polys_idx(classify_logits, attention_mask, threshold):
    classify_sigmoid = (classify_logits[0][1:].sigmoid() > threshold).int()
    classify_preds_ids = []
    for i,classify_preds_id in enumerate(classify_sigmoid):
        if classify_preds_id.sum() > 0:
            classify_preds_ids.append(i)
    return [[classify_preds_ids]]


def select_instances_polys_idx_fusion(classify_preds):
    classify_preds_ids = []
    for i,classify_preds_id in enumerate(classify_preds):
        if "other" not in classify_preds_id:
            classify_preds_ids.append(i)
    return [[classify_preds_ids]]


def decode_merge_logits(merger_logits, valid_items_polys_idxes, classify_logits, vocab):
    batch_len = [len(t[0]) for t in valid_items_polys_idxes]
    batch_items_idx = []
    batch_items_scores = []
    for batch_i, logit in enumerate(merger_logits):
        proposal_scores = [[[], []] for _ in range(batch_len[batch_i])] # [idx, idx_score]
        valid_logit = logit[:batch_len[batch_i], :batch_len[batch_i]]
        # select specific classes for merge decode
        yx = torch.nonzero(valid_logit > 0)
        for y, x in yx:
            score_relitive_idx = y
            score_real_idx = valid_items_polys_idxes[batch_i][0][score_relitive_idx]
            proposal_scores[x][0].append(score_real_idx)
            proposal_scores[x][1].append(valid_logit[y, x])
        items, score = nms(proposal_scores, cal_score='mean')
        batch_items_idx.append(items)
        batch_items_scores.append(score)
    return batch_items_idx,batch_items_scores

def nms(proposal_scores, cal_score='mean'):
    proposals = []
    confidences = []
    for p_s in proposal_scores:
        if len(p_s[0]) > 0:
            if cal_score == 'mean':
                score = torch.tensor(p_s[1]).sigmoid().mean()
            else: # multify
                score = torch.tensor(p_s[1]).sigmoid().prod()
            if p_s[0] not in proposals:
                proposals.append(p_s[0])
                confidences.append(score)
            else:
                idx = proposals.index(p_s[0])
                confidences[idx] = max(confidences[idx], score)
    # nms
    unique_proposal_confidence = list(zip(proposals, confidences))
    sorted_proposals_confidence = sorted(unique_proposal_confidence, key=itemgetter(1), reverse=True)
    sorted_proposal = [t[0] for t in sorted_proposals_confidence]
    exist_flag_lst = [True for _ in range(len(sorted_proposal))]
    output_proposals = []
    output_proposals_scores = []
    for pro_i, pro in enumerate(sorted_proposal):
        if exist_flag_lst[pro_i]:
            output_proposals.append(pro)
            output_proposals_scores.append(sorted_proposals_confidence[pro_i][1])
            for pro_j, tmp_pro in enumerate(sorted_proposal[pro_i + 1:]):
                if overlap(pro, tmp_pro):
                    exist_flag_lst[pro_i + pro_j + 1] = False

    return output_proposals,output_proposals_scores

def overlap(lst1, lst2):
    union_len = len(set(lst1 + lst2))
    if union_len == len(lst1) + len(lst2):
        return False
    else:
        return True


def cal_tp_total(batch_pred_lst, batch_gt_lst, device):
    batch_tp_pred_gt_num = []
    for pred_lst, gt_lst in zip(batch_pred_lst, batch_gt_lst):
        pred_len = len(pred_lst)
        gt_len = len(gt_lst)
        tp = 0
        for pred in pred_lst:
            if pred in gt_lst:
                tp += 1
        batch_tp_pred_gt_num.append([tp, pred_len, gt_len])
    batch_tp_pred_gt_num = torch.tensor(batch_tp_pred_gt_num, device=device)
    return batch_tp_pred_gt_num
