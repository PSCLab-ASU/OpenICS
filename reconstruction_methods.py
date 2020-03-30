import numpy as np
import scipy.io as sio
import torch
from torch import nn
from istanet import ISTANetModel


def reconstruction_method(reconstruction, specifics):
    # a function to return the reconstruction method with given parameters.
    # It's a class with two methods: initialize and run.
    if reconstruction == "ISTANet":
        return ISTANet(specifics)


class ISTANet:
    def __init__(self, specifics):
        # do the initialization of the network with given parameters.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.specifics = specifics
        self.model = ISTANetModel(specifics["layer_num"])
        self.optimizer = None
        self.dataset = None
        self.sensing = None
        self.Phi = None
        self.Qinit = None

    # def initialize(self, dataset, sensing):
    def initialize(self, sensing):
        # do the preparation for the running.
        # init optimizers if training
        # testing is one time forward
        # self.dataset = dataset
        self.sensing = sensing

        self.model = nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.specifics["learning_rate"]
        )

        # Phi is their sampling matrix, should this be involved in the sensing methods section instead?
        Phi_data = sio.loadmat(f"models/phi_0_{self.specifics['cs_ratio']}_1089.mat")
        Phi_input = Phi_data["phi"]
        Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
        self.Phi = Phi.to(self.device)
        # print(self.Phi.size())

        # setup qinit
        Training_data_Name = "Training_Data.mat"
        Training_data = sio.loadmat("models/Training_Data.mat")
        Training_labels = Training_data["labels"]

        X_data = Training_labels.transpose()
        Y_data = np.dot(Phi_input, X_data)
        Y_YT = np.dot(Y_data, Y_data.transpose())
        X_YT = np.dot(X_data, Y_data.transpose())
        self.Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))

    def run(self, stage):
        if stage == "training":
            start_epoch = self.specifics["start_epoch"]
            end_epoch = self.specifics["end_epoch"]
            if start_epoch > 0:
                self.model.load_state_dict(
                    torch.load(f"models/istanet_params_{start_epoch}.pkl")
                )

            for epoch in range(start_epoch + 1, end_epoch + 1):
                for data in self.sensing:
                    batch_x = data
                    batch_x = batch_x.to(self.device)
                    print("PHI SIZE AT MULT:", self.Phi.size())
                    Phix = torch.mm(batch_x, torch.transpose(self.Phi, 0, 1))
                    print(
                        "PHIX AT CREATION",
                        Phix.size(),
                        "PHI SIZE AFTER MULT:",
                        self.Phi.size(),
                    )
                    [x_output, loss_layers_sym] = self.model.forward(Phix, self.Phi, self.Qinit)

                    # Compute and print loss
                    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

                    loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
                    for k in range(self.specifics["layer_num"] - 1):
                        loss_constraint += torch.mean(
                            torch.pow(loss_layers_sym[k + 1], 2)
                        )

                    gamma = torch.Tensor([0.01]).to(self.device)

                    # loss_all = loss_discrepancy
                    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

                    # Zero gradients, perform a backward pass, and update the weights.
                    self.optimizer.zero_grad()
                    loss_all.backward()
                    self.optimizer.step()

                    output_data = (
                        "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n"
                        % (
                            epoch,
                            end_epoch,
                            loss_all.item(),
                            loss_discrepancy.item(),
                            loss_constraint,
                        )
                    )
                    print(output_data)

                    if epoch % 5 == 0:
                        torch.save(
                            self.model.state_dict(), f"models/net_params_{epoch}.pkl"
                        )  # save only the parameters
