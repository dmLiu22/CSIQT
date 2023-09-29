
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

from .iqa_dataset import *
from .samplers import SubsetRandomSampler, IQAPatchDistributedSampler

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import str_to_pil_interp


def build_transform(is_train, config):
    if config.DATA.DATASET == "koniq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "livec":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "live":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "tid2013":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "csiq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    elif config.DATA.DATASET == "kadid":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            # Test transforms
        else:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "spaq":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    if config.DATA.DATASET == "livefb":
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
    transform_t = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((512, 512)),
            transforms.RandomCrop(size=config.DATA.CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return [transform,transform_t]


def build_IQA_dataset(config):
    print(config.DATA.DATASET)
    if config.DATA.DATASET == "koniq":
        train_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "livec":
        train_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "live":
        train_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "tid2013":
        train_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "csiq":
        train_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "kadid":
        train_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "spaq":
        train_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    elif config.DATA.DATASET == "livefb":
        train_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config)[0],
            transform_t=build_transform(is_train=True, config=config)[1],
        )
        test_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config)[0],
            transform_t=build_transform(is_train=False, config=config)[1],
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")

    return train_dataset, test_dataset


def IQA_build_loader(config):
    config.defrost()
    dataset_train, dataset_val = build_IQA_dataset(config=config)
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = IQAPatchDistributedSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
