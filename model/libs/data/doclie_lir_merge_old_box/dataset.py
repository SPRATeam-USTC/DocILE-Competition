import os
import cv2
import tqdm
import torch
import numpy as np
from PIL import Image


class InvalidFormat(Exception):
    pass

class Dataset:
    def __init__(self, loaders, transforms):
        self.loaders = loaders
        self.transforms = transforms

    def get_info(self, idx):
        info = self.loaders[idx]
        image_path = info['image_name']
        image = np.array(Image.open(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        return info, image

    def __len__(self):
        pass

    def __getitem__(self, idx):
        try:
            info, image = self.get_info(idx)
            image, bboxes, label_ids, sentence_embeddings, instance_label, instance_class, line_item_label, image_info = self.transforms(info, image)
            return dict(
                image=image,
                input_embeds=sentence_embeddings,
                bboxes=bboxes,
                labels=label_ids,
                instance_label=instance_label,
                instance_class=instance_class,
                line_item_label=line_item_label,
                image_info=image_info
            )
        except Exception as e:
            print('Error occured while load data_%d: %s' % (idx, e))
            raise e
        

class DataCollator:

    def __call__(self, batch_data):

        def merge1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = pad_id.repeat(len(tensors), max(lengths), 1)
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

        def merge3d(tensors, pad_id):
            dim1 = max([s.shape[0] for s in tensors])
            dim2 = max([s.shape[1] for s in tensors])
            dim3 = max([s.shape[2] for s in tensors])
            out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
            return out
            
        def mask1d(tensors, pad_id):
            lengths= [len(s) for s in tensors]
            out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
            for i, s in enumerate(tensors):
                out[i,1:len(s)] = 1
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

        image = merge3d([torch.from_numpy(data["image"].transpose(2,0,1).astype(np.float32)) for data in batch_data], 0)  
        input_embeds = merge2d([data['input_embeds'] for data in batch_data], 0)    
        attention_mask = mask1d([data['input_embeds'] for data in batch_data], 0)
        input_bboxes = merge2d([torch.from_numpy(data['bboxes']) for data in batch_data], 0)
        pad_id = np.full((1,19), -100).astype('int32')
        labels = merge1d([torch.from_numpy(data['labels']) for data in batch_data], torch.from_numpy(pad_id))
        batch_instance_label = [data['instance_label'] for data in batch_data]
        batch_instance_class = [data['instance_class'] for data in batch_data]
        batch_line_item_label = [data['line_item_label'] for data in batch_data]
        batch_image_info = [data['image_info'] for data in batch_data]

        return {
            "image":image,
            "inputs_embeds":input_embeds,
            "attention_mask":attention_mask,
            "bbox":input_bboxes,
            "labels":labels,
            "instance_label_batch":batch_instance_label,
            "instance_class_batch":batch_instance_class,
            "line_item_label_batch":batch_line_item_label,
            "image_infos":batch_image_info,
            "return_dict":False
        }