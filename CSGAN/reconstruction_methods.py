import torch
import torchvision
from torch import nn


def reconstruction_method(reconstruction,specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if (reconstruction == 'csgan' or reconstruction == 'CSGAN'):
        CSGAN = csgan(specifics=specifics)
        return CSGAN

class csgan():
    def __init__(self, specifics):
        """Constructs the module.

        Args:
          discriminator: The discriminator network. A sonnet module. See `nets.py`.
          generator: The generator network. A sonnet module. For examples, see
            `nets.py`.
          num_z_iters: an integer, the number of latent optimisation steps.
          z_step_size: an integer, latent optimisation step size.
          z_project_method: the method for projecting latent after optimisation,
            a string from {'norm', 'clip'}.
          optimisation_cost_weight: a float, how much to penalise the distance of z
            moved by latent optimisation.
        """

        self._discriminator = MLPMetricNet()
        self._generator = MLPGeneratorNet()
        self._loss = torch.nn.MSELoss()
        self.num_z_iters = specifics['num_z_iters']
        self.z_project_method = specifics['z_project_method']
        self.z_step_size = specifics['z_step_size']
        self.batch_size = specifics['batch_size']
        self.gen_lr = specifics['gen_lr']

    def initialize(self,dset,sensing_method,specifics):
        # do the preparation for the running.
        self.sensing_method = sensing_method
        self.dataset = dset

    def run(self, stage):
        # run the training/testing. print the result.
        if (stage == 'training'):
            # pre-process training data
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ])),
                batch_size=self.batch_size, shuffle=True)

            x_true = self.dataset
            # the discriminator is equivalent to the sensing method in cs_gans
            y_measured = self._discriminator(x_true)

            import torch.optim as optim
            # create your optimizer
            genOptimizer = optim.SGD(self._generator.parameters(), lr=self.gen_lr)
            for batch_idx, (y_measured, x_true) in enumerate(train_loader):
                genOptimizer.zero_grad()  # zero the gradient buffers
                output = self._generator(y_measured)  # forward operation is called, get the output by putting it through current net
                loss = self._loss(output, x_true)  # calculate the loss from however we defined it (in this case criterion is a nn.MSELoss() object) ex. difference between output and target
                loss.backward()  # calculate the gradients of the loss, which ones contributed most to the discrepency of the output and target
                genOptimizer.step()  # Does the update, which is just the for loop above that erfroms f.data.sub_(f.grad.data * learning_rate) on each parameter of net

                discOut = self._discriminator(output)
            print("TODO: display results")
            return 1

        elif (stage == 'testing'):
            # pre-process training data
            x_test = self.dataset
            y_measured = self.sensing_method(x_test)
            self.generator = torch.load('./savedModels/csgan1')  # load the model you want to test
            self.generator.eval()
            x_hat = self.generator(y_measured)

            return x_hat


# not used in testing, only used in training
class MLPMetricNet(nn.Module):
  """Discriminator"""

  def __init__(self, num_outputs=2, name='mlp_metric'):
    super(MLPMetricNet, self).__init__(name=name)
    self.linear = nn.Linear(500, 500, num_outputs, bias=True)

  def forward(self, inputs):
    output = self.linear(torch.flatten(inputs))
    return output


# This is the generator net (incomplete)
class MLPGeneratorNet(nn.Module):
  """MNIST generator net."""

  def __init__(self, name='mlp_generator'):
    super(MLPGeneratorNet, self).__init__(name=name)
    self.linear = nn.Linear(500, 500, 784, bias=True)

  def forward(self, inputs):
    out = self.linear(inputs)
    out = torch.nn.tanh(out)
    return torch.reshape(out, [28, 28, 1])
