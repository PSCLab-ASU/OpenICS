import torchvision
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.vision import VisionDataset
import numpy as np
import skimage.measure as skim
from skimage import io, transform
import torch
import os
import shutil
import pandas 
import matplotlib.pyplot as plt
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
def generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics):
    trans = PreprocessTransform()
    if not os.path.exists(specifics['dataset']):
        print("Dataset folder given in specifics does not exist. Attempting to create dataset from source folder...")
        createDataset(specifics)
    if stage == "testing":
        if (not os.path.exists(specifics['dataset']+"/test")):
            print("Test set could not be found. Please make sure the test set is in the test subfolder.")
            exit()
        test_dataset = CustomDataset(specifics["dataset"]+"/test",transform=transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize(input_width),
                                        transforms.CenterCrop(input_width),
                                        transforms.ToTensor(),
                                        transforms.Lambda(trans)                                            
                                    ]))
        return test_dataset
    elif stage == "training":
        if (not os.path.exists(specifics['dataset']+"/val")):
            print("Validation set could not be found. Please make sure the val set is in the val subfolder.")
            exit()
        if (not os.path.exists(specifics['dataset']+"/train")):
            print("Train set could not be found. Please make sure the train set is in the train subfolder.")
            exit()
        train_dataset = CustomDataset(specifics["dataset"]+"/train",transform=transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize(input_width),
                                        transforms.CenterCrop(input_width),
                                        transforms.ToTensor(),
                                        transforms.Lambda(trans)                                            
                                    ]))
        val_dataset = CustomDataset(specifics["dataset"]+"/val",transform=transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.Resize(input_width),
                                        transforms.CenterCrop(input_width),
                                        transforms.ToTensor(),
                                        transforms.Lambda(trans)                                            
                                    ]))
        return (train_dataset, val_dataset)
        
        
def createDataset(specifics):
    
    if (os.path.exists(specifics['copy_dataset_source_folder']) and  os.path.exists(specifics['copy_dataset_source_folder']+"/train") and  os.path.exists(specifics['copy_dataset_source_folder']+"/test")):
        from pathlib import Path
        Path(specifics['dataset']+"/train").mkdir(parents=True, exist_ok=True)
        Path(specifics['dataset']+"/val").mkdir(parents=True, exist_ok=True)
        Path(specifics['dataset']+"/test").mkdir(parents=True, exist_ok=True)
        datapaths = os.listdir(specifics["copy_dataset_source_folder"]+"/train")
        np.random.shuffle(datapaths)
        cutoff = len(datapaths) - specifics['n_Val_Images']
        for i in range (len(datapaths)):
            filename = str(datapaths[i])
            source = specifics["copy_dataset_source_folder"]+"/train/" + filename
            if (i < cutoff):
                shutil.copyfile(source,specifics['dataset'] + "/train/"+ filename)
            else:
                shutil.copyfile(source,specifics['dataset'] + "/val/"+ filename)
        datapaths = os.listdir(specifics["copy_dataset_source_folder"]+"/test")
        for i in range (len(datapaths)):
            filename = str(datapaths[i])
            source = specifics["copy_dataset_source_folder"]+"/test/" + filename
            shutil.copyfile(source,specifics['dataset'] + "/test/"+ filename)
    else:
        print("Could not find source folder to create datasets from. Please make sure the source folder has both test and train subfolders")
        exit()

    return
class PreprocessTransform: 
    def __call__(self, x):
        x = x*2-1
        return x
def postprocess(x):
    x = (x+1)/2
    return x
def addNoise(clean, sigma):
    clean = torch.tensor(clean)
    noise_vec = torch.rand(clean.shape)
    noise_vec = sigma * np.reshape(noise_vec, newshape=clean.shape)
    noisy = clean + noise_vec
    return noisy
def compute_average_psnr(img,img_hat):
    sz=img.size(0)
    img = img.cpu()
    img_hat = img_hat.cpu().detach()
    return sum([skim.compare_psnr(img[i,:,:,:].numpy()/2.0+0.5,img_hat[i,:,:,:].numpy()/2.0+0.5,data_range=1.0) for i in range(sz)])/sz

def make_prior(num_latents):
    prior_mean = torch.zeros(num_latents, dtype=torch.float32)
    prior_scale = torch.ones(num_latents, dtype=torch.float32)

    return torch.distributions.normal.Normal(prior_mean,prior_scale)

class CustomMnistDataset(Dataset):
    def __init__(self, root_dir, datasetType, transform=None):
        """
        Args:
            type(string): type of the dataset. Can be train, test, or val.
            root_dir (string): Root directory of the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.datasetType = datasetType

    def __len__(self):
        if self.datasetType == "train":
            return 60000
        elif self.datasetType =="val":
            return 0
        elif self.datasetType == "test":
            return 10000
        else:
            return 0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.datasetType,
                                str(idx)+".png")
       # print("img_name:", img_name)
        image = io.imread(img_name)
       # landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
       # sample = {'image': image}

        if self.transform:
            image = self.transform(image)

        return image, 0
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)
def make_dataset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
        #target_dir = os.path.join(directory, target_class)
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = path,0
                instances.append(item)
    return instances

def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
class CustomDataset(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(CustomDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        samples = make_dataset(self.root, extensions =extensions, is_valid_file=is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions


        self.samples = samples
        self.targets = [s[1] for s in samples]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)



