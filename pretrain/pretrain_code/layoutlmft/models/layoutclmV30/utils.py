import torch
import numpy as np
from torch.nn import functional as F


def align_logits(logits):
    batch_size = len(logits)
    max_length = max([_.shape[0] for _ in logits])
    dim = logits[0].shape[1]

    aligned_logits = torch.full((batch_size, max_length, dim), -100, dtype=logits[0].dtype, device=logits[0].device)
    for batch_idx, logits_pb in enumerate(logits):
        aligned_logits[batch_idx, :logits_pb.shape[0]] = logits_pb

    return aligned_logits