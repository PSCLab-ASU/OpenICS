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
        self.dimension = 2 # num of dimensions in each signal

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        if(self.transform):
            self.dset[idx] = self.transform(torch.Tensor(self.dset[idx]), self.dimension)
        return self.dset[idx]

def generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics):
    # a function to generate the corresponding dataset with given parameters. return an instance of the dataset class.
    if(dataset == 'cifar'):
        print("Datset script")
    # pre processing
    elif(type(dataset) == str):
        if(stage == "training" or dataset == "bsd500"):
            n_train_images = specifics['n_train_images']

            # images = glob.glob(dataset + "/*.bmp")
            # dataset = []
            # for img_index, image in enumerate(images):
            #     with open(image, 'rb') as file:
            #         if((img_index + 1) % 1000 == 0):
            #             print("Processed " + str(img_index + 1) + " images")
            #         img = Image.open(file)
            #         np_img = np.array(img).astype(float) / 255
            #         dataset.append(np_img)
            #         if (img_index + 1 == n_train_images):
            #             break;
            # dataset = np.array(dataset)
            # location_and_name = 'Data/BSD500Custom/BSD500-255-FIRST' + str(n_train_images) + ".npy"
            # np.save(location_and_name, dataset)

            # print(n_train_images)
            location_and_name = 'Data/BSD500Custom/BSD500-255-FIRST' + str(n_train_images) + ".npy"
            # location_and_name = 'Data/TrainingData/StandardTestData_256Res.npy'
            dset = NumpyDataset(location_and_name)
        elif(stage == "testing"):
            dset = NumpyDataset(dataset)
            n_test_images = specifics["n_test_images"]

        # preprocess
        # print(dset[:,:,:,:].shape)
        # dset = dset[:,:,:,0]
        dset = np.reshape(dset, (n_train_images, -1, input_width, input_height))
        if(input_channel == 1):
            dset = dset[:,:1,:,:]

        if(stage=='training'):
            dset = dset[:n_train_images, :, :, :].astype(float)
        else:
            # print(dset[:,:,:,0].shape)
            # dset = np.reshape(dset, (-1, input_channel, input_width, input_height))
            dset = dset[:n_test_images, :, :64, :64].astype(float)
        # print(dset.shape)
        # dset = resize(dset, (64, 64))
        # print(dset.shape)
        dset = np.reshape(dset, (-1, input_channel * input_width * input_height))
        # print(dset.shape)

    dset = torch.from_numpy(dset)

    # for i in range(len(dset)):
    #     sample = dset[i]
    #     sample = (np.reshape(sample, (specifics["input_width"], specifics["input_height"], specifics["input_channel"])))
    #     # sample is from 0 to 1 already
    #     # print(i, sample[:,:,0])
    #     plt.imshow(sample)#, cmap='gray')
    #     plt.show()
    #     if (i == 2):
    #         break;
    return dset

def A_handle(A_vals, x):
    (m, n) = A_vals.shape
    (n, BATCH_SIZE) = x.shape

    # A_vals = A_vals.cpu().detach().numpy()
    # x = x.cpu().detach().numpy()
    # A_vals = np.reshape(A_vals, (m, n*channel))
    # x = np.reshape(x, (n*channel, BATCH_SIZE))
    A_vals = A_vals.cpu()
    x = torch.Tensor(x)



    r = torch.matmul(A_vals, x)
    # if channel ==3:
    #     r1 = torch.matmul(A_vals[0, :, :], x[0, :, :]) # (m x n)(n x BATCH_SIZE)
    #     r2 = torch.matmul(A_vals[1, :, :], x[1, :, :])
    #     r3 = torch.matmul(A_vals[2, :, :], x[2, :, :])
    #     r = torch.cat([r1, r2, r3], dim=0)

    # r = r.reshape((channel, m, BATCH_SIZE))
    return r
    #return torch.matmul(A_vals.detach().cpu(), x.clone().detach().cpu())

def At_handle(A_vals, z): # multiply by "transpose" of sensing matrix(an nn.Module)
    A_vals = A_vals.cpu()
    (m, n) = A_vals.shape
    # (m, BATCH_SIZE) = z.shape
    A_vals = np.reshape(A_vals, (n, m)) # transpose matrix

    # if channel == 1:
    r = torch.matmul(A_vals, z)
    # if channel ==3:
    #     r1 = torch.matmul(A_vals[0, :, :], z[0, :, :]) # (m x m)(m x BATCH_SIZE)
    #     r2 = torch.matmul(A_vals[1, :, :], z[1, :, :])
    #     r3 = torch.matmul(A_vals[2, :, :], z[2, :, :])
    #     r = torch.cat([r1, r2, r3])

    # r = r.reshape((BATCH_SIZE, channel * n))
    return r
    # return torch.matmul(torch.t(torch.Tensor(A_vals.type(torch.float).cpu())), z)

# def compute_average_psnr(img,img_hat):
#     sz=img.size(0)
#     return sum([skim.compare_psnr(img[:,:].numpy()/2.0+0.5,img_hat[:,:].numpy()/2.0+0.5,data_range=1.0) for i in range(sz - 1)])/sz
#     # -1 to 1 not 0 to 256

def EvalError(x_hat,x_true):
    # x_hat = x_hat / 255
    # x_true = x_true / 255
    x_hat = torch.Tensor(x_hat).clone()
    x_true = x_true.cpu().clone()
    mse=((x_hat-x_true)**2).mean()#np.mean(np.square(x_hat-x_true),axis=0)
    # xnorm2=np.mean((x_hat - x_true)**2,axis=0)
    mse_thisiter=mse
    nmse_thisiter=1#mse/xnorm2
    psnr_thisiter=10.*torch.log(255*255.0/mse)/np.log(10.)
    return mse_thisiter, nmse_thisiter, psnr_thisiter