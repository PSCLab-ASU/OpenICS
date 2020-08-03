import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.linalg as la


class NumpyDataset(Dataset):
    """Dataset provided by the github repository"""
    def __init__(self, dataset, transform=None):
        self.dset = dataset
        self.transform = transform
        if(self.transform):
            self.dset = self.transform(torch.Tensor(self.dset), self.dimension)
        self.dimension = 2

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        return self.dset[idx]

def generate_dataset(dataset,input_channel,input_width,input_height,stage,specifics):
    previously_trained = specifics["previously_trained"]
    resume = specifics["resume"]

    numpyDataset = np.load(dataset)
    if(stage=='testing'):
        n_val_images = specifics["n_val_images"]
        numpyDataset = numpyDataset[:n_val_images, :, :, :].astype(float)
    elif(resume):
        n_train_images = specifics['n_train_images']
        numpyDataset = numpyDataset[:n_train_images, :, :, :].astype(float)
    else:
        n_train_images = specifics['n_train_images']
        numpyDataset = numpyDataset[:n_train_images, :, :, :].astype(float)

    numpyDataset = np.reshape(numpyDataset, (-1, input_channel * input_width * input_height))

    torchDataset = NumpyDataset(numpyDataset)
    return torchDataset

def A_handle(A_val, x):
    r = torch.matmul(A_val, x)
    return r

def At_handle(A_val, z): # A^H * z
    # A_val = A_val.cpu()
    # A_val = A_val.detach().numpy()
    # A_val = A_val.T.conj()#torch.t(A_val) # for an all-real matrix A.T.conj() is the same as A.T
    # A_val = torch.Tensor(A_val).cuda()
    # r = torch.matmul(A_val, z)

    # TODO SIGNIFICANT CHANGE HERE
    # original backward operator was conj(A_val).transpose
    # np.ortho => opitimal sudo-inverse is the transpose of the np.orth0 matrix
    A_val_inv = la.pinv(A_val.cpu()) # scipy.linalg.pinv()
    A_val_inv = torch.tensor(A_val_inv)
    r = torch.matmul(A_val_inv, z)
    return r

def EvalError(x_hat,x_true):
    x_hat = x_hat
    x_true = x_true
    mse=((x_hat-x_true)**2).mean()
    mse_thisiter=mse
    psnr_thisiter=10.*torch.log(1.0/mse)/np.log(10.)
    return mse_thisiter, psnr_thisiter


"""
Code for Debugging
"""
# for i in range(len(dset)):
#     sample = dset[i]
#     sample = (np.reshape(sample, (specifics["input_width"], specifics["input_height"], specifics["input_channel"])))
#     # sample is from 0 to 1 already
#     # print(i, sample[:,:,0])
#     plt.imshow(sample)#, cmap='gray')
#     plt.show()
#     if (i == 2):
#         break;

# if channel == 1:
# if channel ==3:
#     r1 = torch.matmul(A_vals[0, :, :], z[0, :, :]) # (m x m)(m x BATCH_SIZE)
#     r2 = torch.matmul(A_vals[1, :, :], z[1, :, :])
#     r3 = torch.matmul(A_vals[2, :, :], z[2, :, :])
#     r = torch.cat([r1, r2, r3])


# images = glob.glob(dataset + "/*.bmp")
# dataset = []
# for img_index, image in enumerate(images):
#     with open(image, 'rb') as file:
#         if((img_index + 1) % 1000 == 0):
#             print("Processed " + str(img_index + 1) + " images")
#         img = Image.open(file)
#         np_img = np.array(img).astype(float)
#         dataset.append(np_img)
#         if (img_index + 1 == n_train_images):
#             break;
# dataset = np.array(dataset)
# location_and_name = 'Data/BSD500Custom/BSD500-255-FIRST' + str(n_train_images) + ".npy"
# np.save(location_and_name, dataset)

# location_and_name = 'Data/BSD500Custom/BSD500-255-FIRST' + str(n_train_images) + ".npy"
# location_and_name = 'Data/TrainingData/StandardTestData_256Res.npy'