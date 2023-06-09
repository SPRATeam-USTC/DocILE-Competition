import numpy as np
from . import transform as T
from .dataset import Dataset


def create_dataset(npy_path, image_dirs, config):
    loaders = np.load(npy_path)

    transforms = T.Compose([
        T.CallTokenizedInput(),
        T.CallResizeImage(),
        T.CallDtcTarget(),
        T.CallMlmTarget(mask_embed=np.load(config.mask_embed)[0], is_cover=config.is_cover, mlm_prob=config.mlm_prob),
        T.CallLclTarget(),
        T.CallBDPTarget(bdp_blocks=config.bdp_blocks),
        T.CallMvmTarget(use_mvm=config.use_mvm, mvm_prob=config.mvm_prob/(1-config.mlm_prob))
    ])

    dataset = Dataset(loaders, transforms)
    return dataset