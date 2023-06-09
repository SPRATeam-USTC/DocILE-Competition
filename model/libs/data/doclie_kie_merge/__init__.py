import torch
import numpy as np
from . import transform as T
from torch.utils.data.distributed import DistributedSampler
from .dataset import Dataset, DataCollator

def create_dataset(npy_path, Docliekie_vocab, cfg):
    loaders = np.load(npy_path)

    transforms = T.Compose([
        T.ProcessItems(),
        T.ProcessLabels(Docliekie_vocab),
        T.DirectResize(cfg.resize_type),
        T.LoadBertEmbedding(),
        T.LoadMergerInfo()
    ])

    dataset = Dataset(loaders, transforms)
    return dataset