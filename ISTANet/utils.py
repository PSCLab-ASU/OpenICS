import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import platform
import numpy as np
import copy
import math

import glob
from PIL import Image
import os

def generate_dataset(stage, specifics):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    if(stage == 'testing'):
        return 1


    Training_labels = getTrainingLabels(stage, specifics)

    nrtrain = specifics['nrtrain']
    batch_size = specifics['batch_size']
    if (platform.system() == "Windows"):
        rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                                 shuffle=True)
    else:
        rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                                 shuffle=True)
    return rand_loader


def rgb2ycbcr(rgb):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    return ycbcr.reshape(shape)


# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)


def imread_CS_py(Iorg, specifics):
    block_size = specifics['input_width']
    [row, col] = Iorg.shape
    row_pad = block_size - np.mod(row, block_size)
    col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row / block_size
    col_block = col / block_size
    block_num = int(row_block * col_block)
    img_col = np.zeros([block_size ** 2, block_num])
    count = 0
    for x in range(0, row - block_size + 1, block_size):
        for y in range(0, col - block_size + 1, block_size):
            img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new, specifics):
    block_size = specifics['input_width']
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new - block_size + 1, block_size):
        for y in range(0, col_new - block_size + 1, block_size):
            X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def getTrainingLabels(stage, specifics):
    if (specifics['create_custom_dataset'] == True):
        Training_data = createTrainingLabels(stage, specifics)
    else:
        Training_data = specifics['training_data_fileName']
        if(specifics['training_data_type'] == 'mat'):
            Training_data = sio.loadmat('./%s/%s.mat' % (specifics['data_dir'], Training_data))
            Training_data = Training_data['labels']
        elif(specifics['training_data_type'] == 'npy'):
            Training_data = np.load('./%s/%s.npy' % (specifics['data_dir'], Training_data))
        else:
            raise Exception('training_data_type of ' + specifics['training_data_type'] + ' is unsupported')
    return Training_data

def createTrainingLabels(stage, specifics):
    custom_dataset = './%s/%s.npy' % (specifics['data_dir'], specifics['custom_dataset_name'])
    if(os.path.exists(custom_dataset)):
        return np.load(custom_dataset)

    Training_labels = []
    if(specifics['custom_type_of_image'] == 'bmp'):
        images = glob.glob(specifics['custom_training_data_location'] + '/*.bmp')
    elif(specifics['custom_type_of_image'] == 'tif'):
        images = glob.glob(specifics['custom_training_data_location'] + '/*.tif')
    else:
        raise Exception('custom_type_of_image of ' + specifics['custom_type_of_image'] + ' is unsupported')

    for image in images:
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = np.array(img)
            # convert to grayscale if in RGB
            if (len(img.shape) == 3):
                img = np.mean(img, 2)
            # scale to 1-dimensional vector between 0 and 1
            img = img.reshape((-1)) / 255
            Training_labels.append(img)
    Training_labels = np.array(Training_labels)

    if (not (os.path.exists(specifics['data_dir']))):
        os.mkdir(specifics['data_dir'])
    np.save(os.path.join('./%s/%s.npy' % (specifics['data_dir'], specifics['custom_dataset_name'])), Training_labels)
    print("################################################################\nCreated new file: "
          + './%s/%s.npy' % (specifics['data_dir'], specifics['custom_dataset_name'])
          + "\n################################################################\n")

    return Training_labels

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len