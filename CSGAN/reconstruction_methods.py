import torch
import torchvision
from torch import nn
import torch.optim as optim
import utils


def reconstruction_method(reconstruction,specifics):
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.
    if (reconstruction == 'csgan' or reconstruction == 'CSGAN'):
        CSGAN = csgan(specifics=specifics)
        return CSGAN

class csgan():
    def __init__(self, specifics):
        self.rname='cs-gan'
        self.specifics=specifics

    def initialize(self,dset,sensing_method,specifics):
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
        self.discriminator = MLPMetricNet()
        self.generator = MLPGeneratorNet()
        self.discLoss = nn.CrossEntropyLoss()
        self.genLoss = nn.CrossEntropyLoss()
        self.num_z_iters = specifics['num_z_iters']
        self.z_project_method = specifics['z_project_method']
        self.z_step_size = specifics['z_step_size']
        self.batch_size = specifics['batch_size']
        self.gen_lr = specifics['gen_lr']
        self.sensing_method = sensing_method
        self.dataset = dset
        self.train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('/files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ])),
                batch_size=self.batch_size, shuffle=True)

    def run(self, stage):
        # run the training/testing. print the result.
        if (stage == 'training'):
            # pre-process training data
            x_true = self.dataset
            y_measured = self.discriminator(x_true)
            discrOptimizer = optim.SGD(self.generator.parameters(), lr=self.gen_lr)
            genOptimizer = optim.SGD(self.generator.parameters(), lr=self.gen_lr)
            for iter in range(self.specifics['num_training_iterations']):
                # Train discriminator
                for param in self.generator.parameters(): # freeze the generator
                    param.requires_grad = False
                for epoch in range(self.specifics["discrEPOCHS"]):
                    for i, (img, label) in enumerate(self.train_loader):
                        # real data
                        discrOptimizer.zero_grad()
                        measurement = self.sensing_method(img)
                        prediction = self.discriminator(measurement)
                        discLoss = self.discLoss(prediction, torch.ones((prediction[:, 0]).shape))
                        discLoss.backward()
                        discrOptimizer.step()

                        # fake data
                        discrOptimizer.zero_grad()
                        fake_img = self.generator(utils.addNoise(torch.randn(img.size())), .1/.255)
                        measurement = self.sensing_method(fake_img)
                        prediction = self.discriminator(measurement)
                        discLoss = self.discLoss(prediction, torch.ones((prediction[:, 0]).shape))
                        discLoss.backward()
                        discrOptimizer.step()
                for param in self.generator.parameters(): # unfreeze the generator
                    param.requires_grad = True

                # Train generator
                for param in self.discriminator.parameters(): # freeze the discriminator
                    param.requires_grad = False
                for epoch in range(self.specifics["genEPOCHS"]):
                    for i, (img, label) in enumerate(self.train_loader):
                        genOptimizer.zero_grad()
                        fake_img = self.generator(utils.addNoise(torch.randn(img.size())), .1/.255)
                        measurement = self.sensing_method(fake_img)
                        prediction = self.discriminator(measurement)
                        genLoss = self.genLoss(prediction, torch.ones((prediction[:, 0]).shape))
                        genLoss.backward()
                        genOptimizer.step()
                for param in self.discriminator.parameters(): # unfreeze the discriminator
                    param.requires_grad = True

            print("TODO: display results")
            return 1

        # TODO how will we be testing the cs_gan
        elif (stage == 'testing'):
            # pre-process training data
            with torch.no_grad():
                self.generator.eval()
                val_psnrs = []
                step_size = .0001
                A = torch.randn(self.specifics['m'], self.specifics['n'])
                for img, _ in iter(self.train_loader):
                    # x = img.cuda()
                    # z0 = torch.zeros(img.shape())
                    # y = A(x)
                    # x_hat = self.generator(z0)
                    #
                    # for iter in self.specifics['gradient_desc_iter']:  # gradient descent
                    #     expression = self.generator(z0) - y
                    #     x_hat = x_hat - step_size * A.transpose.matmul(expression)

                    # img_hat = x_hat
                    val_psnrs.append(utils.compute_average_psnr(img.cpu(), img_hat.detach().cpu()))
                val_psnr = sum(val_psnrs) / len(val_psnrs)
                print("average test psnr:" + str(val_psnr))
            return 1


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
