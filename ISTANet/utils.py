import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import platform
import copy
import math
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os


def generate_dataset(input_channel,input_width,input_height, stage, specifics):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.

    Training_labels = getTrainingLabels(input_channel,input_width,input_height, stage, specifics)

    nrtrain = specifics['nrtrain']
    batch_size = specifics['batch_size']
    if (platform.system() == "Windows"):
        rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                                 shuffle=True)
    else:
        rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                                 shuffle=True)
    return rand_loader, Training_labels


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
    if(specifics['sudo_rgb'] == True): # X_col is a list of tensors
        X_rec3 = []
        for i in range(3):
            X0_rec = np.zeros([row_new, col_new])

            count = 0
            for x in range(0, row_new - block_size + 1, block_size):
                for y in range(0, col_new - block_size + 1, block_size):
                    X0_rec[x:x + block_size, y:y + block_size] = X_col[i][:, count].reshape([block_size, block_size])
                    # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
                    count = count + 1
            X_rec = X0_rec[:row, :col]
            X_rec3.append(X_rec)
        X_rec = np.array(X_rec3)
    else: # X_col is a tensor
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
    if mse <= 1:
        return 48
    PIXEL_MAX = 255.0

    # 20 * log(MaxPixel) - 10 * log(MSE)

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def getTrainingLabels(input_channel,input_width,input_height, stage, specifics):
    if (specifics['create_custom_dataset'] == True):
        Training_data = createTrainingLabels(stage, specifics)
    else:
        Training_data = specifics['training_data_fileName']
        if specifics['training_data_fileName'] == 'mnist':
            Training_data = datasets.MNIST(root='./data/testDatasetDownload', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(input_width),
                                               transforms.CenterCrop(input_width),
                                               transforms.ToTensor()
                                           ]))
            Training_data = Training_data.data.numpy()
            Training_data = Training_data.reshape((-1, input_channel*input_width*input_height)) / 255
        elif specifics['training_data_fileName'] == 'cifar10':
            Training_data = datasets.CIFAR10(root='./data/testDatasetDownload', train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(input_width),
                                                 transforms.CenterCrop(input_width),
                                                 transforms.ToTensor()
                                             ]))
            Training_data = Training_data.data.numpy()
            Training_data = Training_data.reshape((-1, input_channel*input_width*input_height)) / 255
        elif specifics['training_data_fileName'] == 'celeba':
            Training_data = datasets.CelebA(root='./data/testDatasetDownload', split="train", download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(input_width),
                                                transforms.CenterCrop(input_width),
                                                transforms.ToTensor()
                                            ]))
            Training_data = Training_data.data.numpy()
            Training_data = Training_data.reshape((-1, input_channel*input_width*input_height)) / 255
        # elif specifics['training_data_fileName'] == '/storage-t1/database/cs-framework-database/mnist/train':
        #     Training_data = CustomDataset(specifics['training_data_fileName'], specifics,
        #                                   transform=transforms.Compose([
        #                                        transforms.Resize(input_width),
        #                                        transforms.CenterCrop(input_width),
        #                                        transforms.ToTensor()
        #                                    ]))
        elif(specifics['training_data_type'] == 'mat'):
            Training_data = sio.loadmat('%s/%s.mat' % (specifics['data_dir'], Training_data))
            Training_data = Training_data['labels']
        elif(specifics['training_data_type'] == 'npy'):
            Training_data = np.load('%s/%s.npy' % (specifics['data_dir'], Training_data))
        else:
            raise Exception('training_data_type of ' + specifics['training_data_type'] + ' is unsupported')
    Training_data = Training_data[:specifics['nrtrain']]
    np.random.shuffle(Training_data)

    return Training_data

def createTrainingLabels(stage, specifics):
    custom_dataset = '%s/%s.npy' % (specifics['data_dir'], specifics['custom_dataset_name'])
    if(os.path.exists(custom_dataset)):
        return np.load(custom_dataset)

    Training_labels = []
    if(specifics['custom_type_of_image'] == 'bmp'):
        images = glob.glob(specifics['custom_training_data_location'] + '/*.bmp')
    elif(specifics['custom_type_of_image'] == 'tif'):
        images = glob.glob(specifics['custom_training_data_location'] + '/*.tif')
    elif (specifics['custom_type_of_image'] == 'png'):
        images = glob.glob(specifics['custom_training_data_location'] + '/*.png')
    elif (specifics['custom_type_of_image'] == 'jpg'):
        images = glob.glob(specifics['custom_training_data_location'] + '/*.jpg')
    else:
        raise Exception('custom_type_of_image of ' + specifics['custom_type_of_image'] + ' is unsupported')

    for i, image in enumerate(images):
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = np.array(img)
            if (specifics['sudo_rgb']):
                if (i == (specifics['nrtrain']/3)):
                    break
                # scale to between 0 and 1
                Training_labels.append(img[:, :, 0].reshape((1, specifics['input_width'], specifics['input_width'])) / 255)
                Training_labels.append(img[:, :, 1].reshape((1, specifics['input_width'], specifics['input_width'])) / 255)
                Training_labels.append(img[:, :, 2].reshape((1, specifics['input_width'], specifics['input_width'])) / 255)
            else:
                # scale to between 0 and 1
                img = img.reshape((specifics['input_channel'], specifics['input_width'], specifics['input_width'])) / 255
                Training_labels.append(img)

                # fig3 = plt.figure()
                # plt.imshow(Training_labels[i][0])
                # plt.show()

    Training_labels = np.array(Training_labels)
    Training_labels = np.reshape(Training_labels, (-1, specifics['input_width'] * specifics['input_width']))

    if (not (os.path.exists(specifics['data_dir']))):
        os.mkdir(specifics['data_dir'])
    np.save(os.path.join('%s/%s.npy' % (specifics['data_dir'], specifics['custom_dataset_name'])), Training_labels)
    print("################################################################\nCreated new file: "
          + '%s/%s.npy' % (specifics['data_dir'], specifics['custom_dataset_name'])
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

class CustomDataset(Dataset):
    def __init__(self, root: str, specifics, transform=transforms.ToTensor()):
        samples = []
        srcdir = os.listdir(root)
        for i in range(len(srcdir)):
            filename = str(srcdir[i])
            samples.append(filename)


        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.specifics = specifics
        self.root = root
        self.samples = samples

    def __getitem__(self, index: int):
        path = self.root + '/' + self.samples[index]
        sample = Image.open(path)
        sample = self.transform(sample)
        torch.reshape(sample, (-1, self.specifics['input_width']*self.specifics['input_width']))
        torch.div(sample, 255)
        return sample

    def __len__(self) -> int:
        return len(self.samples)