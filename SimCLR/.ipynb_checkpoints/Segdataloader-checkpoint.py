import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import scipy.io as sio

class BaseDataset(Dataset):
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        # Set image path (subfolders for different modes)
        self.data_root = data_root+"/"+mode+"/images"
        # Limited Mode for each dataset
        self.mode = mode
        assert mode in ["train", "val", "test"]
        # Parsing data path list
        self.data_paths = self._parse_data_list()
        # total number of samples
        self._length = len(self.data_paths)
        # loactaion of labels dicionery
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        # agumentation for segmentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256))
        ])
        


    def __len__(self):
        return self._length


    @staticmethod
    # if mode is train, this fuction will be called
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image
        
        
        
class REFUGE2Base(BaseDataset):
    """REFUGE2 Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """

    def __getitem__(self, i):
        # Read image and mask path
        example = dict((k, self.labels[k][i]) for k in self.labels)
        if self.mode == "val":
            segmentation = Image.open(example["file_path_"].replace("images", "mask").replace('jpg','png')) # same name, different postfix
        else:
            segmentation = Image.open(example["file_path_"].replace("images", "mask").replace('jpg','bmp')) # same name, different postfix
        image = Image.open(example["file_path_"]).convert("RGB")    
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        # if train for this dataset
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        # only support binary segmentation now:
        segmentation = np.array(segmentation).astype(np.float32)

        segmentation[segmentation== 0.] = 1
        segmentation[segmentation== 255.] = 2
        segmentation[segmentation== 128.] = 0
        example["mask"] = torch.Tensor(segmentation)   
        image = np.array(image).astype(np.float32) / 255.
        example["pixel_values"] = torch.Tensor(image)

        return example

    def _parse_data_list(self):

        if self.mode == "train":
            train_imgs = glob.glob(os.path.join(self.data_root, "*.jpg"))
            return train_imgs
        elif self.mode == "val":
            val_imgs = glob.glob(os.path.join(self.data_root, "*.jpg"))
            return val_imgs
        elif self.mode == "test":
            test_imgs = glob.glob(os.path.join(self.data_root, "*.jpg"))
            return test_imgs
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test !")


    
class ORIGAbase(BaseDataset):
    """ORIGA Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """

    def __getitem__(self, i):

        # Read image and mask path
        example = dict((k, self.labels[k][i]) for k in self.labels)
        segmentation = sio.loadmat(example["file_path_"].replace("images", "mask").replace('jpg','mat'))['maskFull']  # same name, different postfix
        # segmentation = Image.fromarray(segmentation_data.astype(np.uint8))        
        
        image = Image.open(example["file_path_"]).convert("RGB")    
        # Resize the image
        if self.size is not None:
            segmentation = Image.fromarray(segmentation.astype(np.uint8)) 
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        # if train for this dataset
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)
        segmentation = np.array(segmentation).astype(np.float32)  
        
        remap = np.array([2, 0, 1])  
        segmentation = remap[segmentation.astype(np.uint8)]

        example["mask"] = torch.Tensor(segmentation)   
        image = np.array(image).astype(np.float32) / 255.
        example["pixel_values"] = torch.Tensor(image)
        return example
    def _parse_data_list(self):
        imgs = glob.glob(os.path.join(self.data_root, "*.jpg"))
        return imgs
class G1020base(BaseDataset):
    """G1020 Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    

    def __getitem__(self, i):
        
        # Read image and mask path
        example = dict((k, self.labels[k][i]) for k in self.labels)
        segmentation = Image.open(example["file_path_"].replace("images", "mask").replace('jpg','png'))
        image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix
        # Resize the image
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        # if train for this dataset
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)        
        segmentation = np.array(segmentation).astype(np.float32) 
        remap = np.array([2, 0, 1])  
        segmentation = remap[segmentation.astype(np.uint8)]
        example["mask"] = torch.Tensor(segmentation)   
        image = np.array(image).astype(np.float32) / 255.
        example["pixel_values"] = torch.Tensor(image)
        return example

    def _parse_data_list(self):
        imgs = glob.glob(os.path.join(self.data_root, "*.jpg"))

        return imgs
    
class RIMONEbase(BaseDataset):
    """RIMONEbase Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        segmentation = Image.open(example["file_path_"].replace("images", "mask")).convert("L")
        image = Image.open(example["file_path_"]).convert("RGB")    
        # Resize the image
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)
        # if train for this dataset
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)        
        segmentation = np.array(segmentation).astype(np.float32)
        segmentation[segmentation== 0.] = 1
        segmentation[segmentation== 255.] = 2
        segmentation[segmentation== 128.] = 0
        example["mask"] = torch.Tensor(segmentation)   
        image = np.array(image).astype(np.float32) / 255.
        example["pixel_values"] = torch.Tensor(image)
        return example
    def _parse_data_list(self):
        imgs = glob.glob(os.path.join(self.data_root, "*.png"))
        return imgs