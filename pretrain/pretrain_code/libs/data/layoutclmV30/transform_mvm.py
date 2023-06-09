import cv2
import copy
import random
import numpy as np
import os.path as osp


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data


class CallTokenizedInput:
    def __init__(self, max_length=128):
        self.max_length = max_length

    def __call__(self, info, image, loader, data_idx):
        sentences_embed = np.load(info['sentences_embed'])
        bboxes = np.array(info['bboxes']).astype(np.int64)

        if len(sentences_embed) > self.max_length:
            start_index = random.randint(0, max(0, len(bboxes) - self.max_length - 1))
            input_sentences_embed = sentences_embed[start_index:start_index+self.max_length]
            input_bboxes = bboxes[start_index:start_index+self.max_length]
        else:
            input_sentences_embed = sentences_embed
            input_bboxes = bboxes
        
        # append [CLS] tokens to the first.
        cls_bbox = np.array([0,0,1000,1000]).astype('int64')
        input_bboxes = np.concatenate([cls_bbox[None, :], input_bboxes], axis=0)
        cls_embed = np.zeros_like(sentences_embed[0])
        input_sentences_embed = np.concatenate([cls_embed[None, :], input_sentences_embed], axis=0)

        return info, image, input_sentences_embed, input_bboxes, loader, data_idx


class CallResizeImage:
    '''
        Resize input image to a small size to save GPU Memory.
    '''
    def __init__(self, image_H=512, image_W=512, max_wh=512):
        self.image_H = image_H
        self.image_W = image_W
        self.max_wh = max_wh
    
    def __call__(self, info, image, input_sentences_embed, input_bboxes, loader, data_idx):
        H, W, _ = image.shape
        ratio_H = self.image_H / H
        ratio_W = self.image_W / W
        image = cv2.resize(image, dsize=(self.image_W, self.image_H), interpolation=cv2.INTER_LINEAR)
        input_bboxes[:, 0::2] = input_bboxes[:, 0::2] * ratio_W
        input_bboxes[:, 1::2] = input_bboxes[:, 1::2] * ratio_H
        input_bboxes = input_bboxes.clip(0, self.max_wh)
        return info, image, input_sentences_embed, input_bboxes, loader, data_idx


class CallDtcTarget:
    '''
    Generate the labels of documnet type from image path.
    It's worth noting that the information of documnet type is contained in the image path.
    '''
    def __call__(self, info, image, input_sentences_embed, input_bboxes, loader, data_idx):
        try:
            dtc_labels = int(osp.basename(osp.dirname(info['image_name'])))
        except:
            dtc_labels = -100
        return image, input_sentences_embed, input_bboxes, dtc_labels, loader, data_idx


class CallMlmTarget:
    '''
        Generate the mask of input_ids for MLM task.
    '''
    def __init__(self, mask_embed, mlm_prob=0.15, random_token_prob=0.1, leave_unmasked_prob=0.1):
        self.mask_embed = mask_embed
        self.mlm_prob = mlm_prob
        self.random_token_prob = random_token_prob
        self.leave_unmasked_prob = leave_unmasked_prob
    
    def __call__(self, image, input_sentences_embed, input_bboxes, dtc_labels, loader, data_idx):
        sz = len(input_sentences_embed)
        mask = np.random.rand(sz) < self.mlm_prob
        mask[0] = False # remove [CLS]
        if not mask.any(): # select none
            mask[random.randint(1, len(mask) - 1)] = True # random select one token as mask input ids
        
        # create target
        unmask_sentences_embed = copy.deepcopy(input_sentences_embed)
        mlm_mask = copy.deepcopy(mask).astype(np.int64)

        # creat source
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        # mask sentence
        if mask is not None:
            input_sentences_embed[np.where(mask==True)[0]] = self.mask_embed

        # rand selecet
        if rand_mask is not None:
            rand_ids = np.where(rand_mask==True)[0]
            for idx in rand_ids:
                info = loader.get_info_only(loader.get_diff_idx(data_idx))
                proposal_sentences_embed = np.load(info['sentences_embed'])
                input_sentences_embed[idx] = proposal_sentences_embed[np.random.choice(len(proposal_sentences_embed))]
        
        return image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, dtc_labels


class CallLclTarget:
    '''
        Generate the labels for Language Contrastive Learning (LCL) task.
    '''
    def __call__(self, image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, dtc_labels):
        lcl_labels = np.where(mlm_mask==True)[0] # target for language Contrastive learning
        return image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, lcl_labels, dtc_labels
        

class CallBDPTarget:
    def __init__(self, image_H=512, image_W=512, bdp_blocks=4):
        self.image_H = image_H
        self.image_W = image_W
        self.num_blocks = bdp_blocks

    '''
    Generate the labels of paried bboxes direction for Boxes Direction Prediction (BDP) task.
    '''
    def __call__(self, image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, lcl_labels, dtc_labels):
        xc, yc = np.split(input_bboxes.reshape(-1,2,2).mean(1), 2, axis=1) # xc yc
        bdp_labels = np.full(xc.shape[0], -100).astype('int64')
        div_H = self.image_H // self.num_blocks
        div_W = self.image_W // self.num_blocks
        for row_id in range(self.num_blocks):
            for col_id in range(self.num_blocks):
                min_H = div_H * row_id
                max_H = div_H * (row_id + 1)
                min_W = div_W * col_id
                max_W = div_W * (col_id + 1)
                label = col_id + self.num_blocks * row_id
                condition_1 = yc >= min_H
                condition_2 = yc < max_H
                condition_3 = xc >= min_W
                condition_4 = xc < max_W
                condition = np.logical_and(np.logical_and(condition_1, condition_2), \
                    np.logical_and(condition_3, condition_4))[:, 0]
                bdp_labels[condition] = label

        return image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, lcl_labels, dtc_labels, bdp_labels


class CallMvmTarget:
    '''
        Generate the labels for Masked Vision Model (MVM) task
    '''
    def __init__(self, mvm_prob=0.088):
        self.mvm_prob = mvm_prob # rel_mvm_prob = mvm_prob * (1-mlm_prob)

    def __call__(self, image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, lcl_labels, dtc_labels, bdp_labels):
        sz = len(input_sentences_embed)
        mask = np.random.rand(sz) < self.mlm_prob
        mask = np.logical_and(mask, ~mlm_mask) # remove mlm tokens
        mask[0] = False # remove [CLS]
        if not mask.any(): # select none
            select_mask = ~mlm_mask
            select_mask[0] == False
            select_range = np.where(select_mask==True)[0] # avail mask for masked vision model
            mask[np.random.shuffle(select_range)[0]] = True # random select one token as mask input ids
        
        unmask_image = copy.deepcopy(image)
        cover_bboxes = input_bboxes[mask]
        for bbox in cover_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, lcl_labels, dtc_labels, bdp_labels, unmask_image, mask