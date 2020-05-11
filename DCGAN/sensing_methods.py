import torch
from torch.utils.data import DataLoader


class Sensor:
    def __init__(self, method, reconstruction, specifics, m, n):
        self.method = method
        self.reconstruction = reconstruction
        self.specifics = specifics

        if self.method == "Gaussian" and self.reconstruction != "ISTANet":
            self.A = torch.randn(m, n)
            self.A.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def sensing_method(self, data):
        if self.method == "Gaussian" and self.reconstruction == "ISTANet":
            return DataLoader(
                dataset=data, batch_size=self.specifics["batch_size"], shuffle=True
            )
        elif self.method == "Gaussian":
            return self.A * data
