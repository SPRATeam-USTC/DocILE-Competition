import os
import cv2
import torch
import random
import numpy as np
from detectron2.structures import ImageList

class InvalidFormat(Exception):
    pass

class Dataset:
    def __init__(self, emb_npy, transforms):
        self.emb_npy = emb_npy
        self.transforms = transforms

    def _get_data(self, idx):
        info = self.emb_npy[idx]
        image_name = info['image_name']
        if os.path.exists(image_name):
            image = cv2.imread(image_name)
        else:
            image = cv2.imread(os.path.join(self.image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return info, image

    def get_info(self, idx):
        lines = self.emb_npy[idx]['tokens']
        return len(lines)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def __getitem__(self, idx):
        try:
            info, image = self._get_data(idx)
            image, input_sentences_embed, input_bboxes, unmask_sentences_embed, mlm_mask, \
                lcl_labels, dtc_labels, bdp_labels, unmask_image, mvm_mask = self.transforms(info, image, self.emb_npy, idx)
            return dict(
                image=image,
                input_sentences_embed=input_sentences_embed,
                input_bboxes=input_bboxes,
                unmask_sentences_embed=unmask_sentences_embed,
                mlm_mask=mlm_mask,
                lcl_labels=lcl_labels,
                dtc_labels=dtc_labels,
                bdp_labels=bdp_labels,
                unmask_image=unmask_image,
                mvm_mask=mvm_mask
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx)
            raise e
        

class DataCollator:

    def __call__(self, batch_data):

        def merge1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = s
            return out
    
        def merge2d(tensors, pad_id):
            dim1 = max([s.shape[0] for s in tensors])
            dim2 = max([s.shape[1] for s in tensors])
            out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i, :s.shape[0], :s.shape[1]] = s
            return out

        def mask1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,:len(s)] = 1
            return out

        def merge_sentences(batch_sentences, pad_id):
            dim1 = len(batch_sentences)
            dim2 = max([len(_) for _ in batch_sentences])
            dim3 = min(self.max_sentence, max([len(_) for sentences in batch_sentences for _ in sentences])) # truncate to the max length
            merged_sentences = np.full((dim1, dim2, dim3), pad_id)
            masks = np.zeros((dim1, dim2, dim3))
            for batch_idx, sentences in enumerate(batch_sentences):
                for sentence_idx, sentence in enumerate(sentences):
                    merged_sentences[batch_idx, sentence_idx, :min(dim3, len(sentence))] = np.array(sentence)[:min(dim3, len(sentence))]
                    masks[batch_idx, sentence_idx, :min(dim3, len(sentence))] = 1
            return merged_sentences, masks

        image = ImageList.from_tensors([torch.from_numpy(data["image"].transpose(2,0,1)) for data in batch_data], 32)
        unmask_image = ImageList.from_tensors([torch.from_numpy(data["image"].transpose(2,0,1)) for data in batch_data], 32)
        input_sentences_embed = merge2d([torch.from_numpy(data['input_sentences_embed']) for data in batch_data], 0)
        unmask_sentences_embed = merge2d([torch.from_numpy(data['unmask_sentences_embed']) for data in batch_data], 0)
        attention_mask = mask1d([torch.from_numpy(data['input_sentences_embed']) for data in batch_data], 0)
        input_bboxes = merge2d([torch.from_numpy(data['input_bboxes']) for data in batch_data], 0)
        mlm_masks = merge1d([torch.from_numpy(data['mlm_mask']) for data in batch_data], 0)
        mvm_masks = merge1d([torch.from_numpy(data['mvm_mask']) for data in batch_data], 0)
        lcl_labels = merge1d([torch.from_numpy(data['lcl_labels']) for data in batch_data], -100)
        dtc_labels = torch.from_numpy(np.array([data['dtc_labels'] for data in batch_data])).long()
        bdp_labels = merge1d([torch.from_numpy(data['bdp_labels']) for data in batch_data], -100)

        return {
            "image":image,
            "unmask_image":unmask_image,
            "inputs_embeds":input_sentences_embed,
            "unmask_embed":unmask_sentences_embed,
            "attention_mask":attention_mask,
            "bbox":input_bboxes,
            "mlm_masks":mlm_masks,
            "mvm_masks":mvm_masks,
            "lcl_labels":lcl_labels,
            "dtc_labels":dtc_labels,
            "bdp_labels":bdp_labels,
            "return_dict":False
        }