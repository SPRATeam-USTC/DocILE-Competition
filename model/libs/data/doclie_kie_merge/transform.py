import os
import cv2
import copy
import random
import numpy as np
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data


class ProcessItems:
    '''
        Items will no appear when classify
    '''
    def __init__(self):
        pass
    
    def __call__(self, info, image):
        H, W, _ = image.shape
        input_bboxes = [info['polys'][i] for i in range(len(info['polys']))]
        input_bboxes = np.array(input_bboxes)
       
        return info, image, input_bboxes

class ProcessLabels:
    '''
        Transform raw labels to corresponding dict_id
    '''
    def __init__(self, vocab):
        self.vocab = vocab
        pass
    
    def __call__(self, info, image, input_bboxes):
        labels = info['labels']
        num_class = len(self.vocab)
        onehot_labels = np.zeros((len(labels), num_class), dtype=np.int32)
        for i,label in enumerate(labels):
            if label == ["other"]:
                continue
            for label_li in label:
                label_id = self.vocab.word_to_id(label_li.lower())
                onehot_labels[i][label_id] = 1
            
        label_ids = onehot_labels
        
        return info, image, input_bboxes, label_ids

class Poly2BBox:
    '''
        Transform polys to bboxes. x1y1x2y2x3y3x4y4 -> x1y1x2y2, shape: (num, 4, 2) -> (num, 4)
    '''
    def __init__(self):
        pass
    
    def __call__(self, info, image, input_polys, label_ids):
        input_polys = np.array(input_polys)
        x1 = input_polys[:, :, 0].min(-1)
        y1 = input_polys[:, :, 1].min(-1)
        x2 = input_polys[:, :, 0].max(-1)
        y2 = input_polys[:, :, 1].max(-1)
        input_bboxes = np.stack([x1, y1, x2, y2], axis=-1)
        return info, image, input_bboxes, label_ids


class DirectResize:
    '''
        Resize input image to a fixed size
    '''
    def __init__(self, resize_type):
        assert resize_type in ['fixed', 'half', 'none']
        self.resize_type = resize_type
    
    def __call__(self, info, image, input_bboxes, label_ids):
        if self.resize_type == 'none':
            target_h, target_w, _ = image.shape
            target_max = np.array([target_w, target_h, target_w, target_h])[None, :]
            input_bboxes = np.clip(input_bboxes, 0, target_max)
            return info, image, input_bboxes, label_ids

        if self.resize_type == 'fixed':
            target_h, target_w = 512, 512
        else:
            ori_h, ori_w, _ = image.shape
            target_h, target_w = int(ori_h / 2), int(ori_w / 2)
        H, W, _ = image.shape
        
        image = cv2.resize(image, dsize=(target_w, target_h))
        input_bboxes[:, 0::2] = input_bboxes[:, 0::2] * target_w
        input_bboxes[:, 1::2] = input_bboxes[:, 1::2] * target_h
        input_bboxes = np.round(input_bboxes).astype('int32')
        target_max = np.array([target_w, target_h, target_w, target_h])[None, :]
        input_bboxes = np.clip(input_bboxes, 0, target_max)
        return info, image, input_bboxes, label_ids

class LoadBertEmbedding:
    '''
        Load bert embedding from disks, and append [CLS] tokens to the first.
    '''
    def __init__(self):
        pass
    
    def __call__(self, info, image, input_bboxes, label_ids):
        emb_path = info['sentences_embed_path']
        foreground_bbox_num = input_bboxes.shape[0]
        sentence_embeddings = torch.load(emb_path)[:foreground_bbox_num]
        
        # append [CLS] tokens to the first
        h, w, _ = image.shape
        cls_bbox = np.array([0, 0, w, h]).astype('int32')
        input_bboxes = np.concatenate([cls_bbox[None, :], input_bboxes], axis=0)
        cls_embed = torch.zeros_like(sentence_embeddings[0])
        sentence_embeddings = torch.cat([cls_embed[None, :], sentence_embeddings], dim=0)
        cls_label = np.full((1,36), -100).astype('int32')
        label_ids = np.concatenate([cls_label, label_ids], axis=0)
        return info, image, input_bboxes, label_ids, sentence_embeddings


class LoadMergerInfo:
    def __init__(self):
        pass
    
    def __call__(self, info, image, input_bboxes, label_ids, sentence_embeddings):
        return image, input_bboxes, label_ids, sentence_embeddings, info["instance_label"], info["instance_class"], info
        
