import torch
import numpy as np
import skimage.measure as skim
from skimage.transform import resize
from torch.utils.data import Dataset
import glob
from PIL import Image
import matplotlib.pyplot as plt


class NumpyDataset(Dataset):
    """Dataset provided by the github repository"""
    def __init__(self, root, transform=None):
        self.dset = np.load(root)
        self.transform = transform

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        if self.transform:
            self.dset = self.transform(self.dset)
        return self.dset[idx]

# class ImagesDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform
#
#     def __len__(self):
#         dset = np.load(self.root)
#         return len(dset)
#
#     def __getitem__(self, idx):
#         dset = np.load(self.root)
#         if self.transform:
#             dset = self.transform(dset)
#         return dset[idx]

def generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    if(dataset == 'cifar'):
        print("Datset script")

    # pre processing
    elif(type(dataset) == str):
        if(stage == "training" or dataset == "bsd500"):
            images = glob.glob(dataset + "/*.bmp")
            dataset = []
            num_of_imgs = 0
            for image in images:
                with open(image, 'rb') as file:
                    num_of_imgs += 1
                    if(num_of_imgs % 1000 == 0):
                        print("Processed " + str(num_of_imgs) + " images")
                    img = Image.open(file)
                    np_img = np.array(img)
                    dataset.append(np_img)
                    if (num_of_imgs == specifics["n_Train_Images"] or num_of_imgs == 1000):
                        break;

            dataset = np.array(dataset)
            np.save('Data/BSD500Custom/BSD500FIRST' + str(num_of_imgs) + ".npy", dataset)
            # dataset = torch.tensor(dataset)
            # print(dataset.shape)
            # yields (torch.Size([1000, 64, 64, 3]))

            dset = NumpyDataset(dataset)
            preprosdset = resize(dset, (num_of_imgs - 1, 64, 64), anti_aliasing=True)
        elif(stage == "testing"):
            dset = NumpyDataset(dataset)
            preprosdset = dset[:,0,:,:].astype(float)
            preprosdset = resize(preprosdset, (7, 64, 64), anti_aliasing=True)

        # print(preprosdset.shape)
        preprosdset = np.transpose(np.reshape(preprosdset, (input_channel * input_width * input_height, -1)))
        # print(preprosdset.shape)
        # for i in range(len(preprosdset)):
        #     sample = preprosdset[i]
        #     print(i, sample.shape)
        #     plt.imshow(sample, cmap='gray')
        #     plt.show()
    preprosdset = torch.from_numpy(preprosdset)
    return preprosdset

def A_handle(A_vals, x):
    return torch.matmul(torch.Tensor(A_vals).cpu(), x.clone().detach().cpu())

def At_handle(A_vals_tf, z): # will not support non-real numbers
    return torch.matmul(torch.t(torch.Tensor(A_vals_tf)), z.clone().detach().cpu())

def generateAVal(m,n):
    return np.float32(1.0 / np.sqrt(float(m)) * np.random.randn(m, n))

def compute_average_psnr(img,img_hat):
    sz=img.size(0)
    return sum([skim.compare_psnr(img[:,:].numpy()/2.0+0.5,img_hat[:,:].numpy()/2.0+0.5,data_range=1.0) for i in range(sz - 1)])/sz