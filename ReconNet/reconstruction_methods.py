import torch
import torchvision
import utils as u
class reconstruction_method():
    def __init__(self,reconstruction,specifics):
        self.rname='reconnet'
        self.specifics=specifics
    def initialize(self,dset,sensing_method):
        self.net = u.reconnet(self.specifics['m'], self.specifics['input_width'], self.specifics['input_channel'])
        self.dataset=dset
        self.sensing=sensing_method
        self.traindataloader = torch.utils.data.DataLoader(self.dataset.train(),
                                                  batch_size=self.specifics["batchsize"], shuffle=True, num_workers=2)
        self.testdataloader=torch.utils.data.DataLoader(self.dataset.test(),
                                                  batch_size=self.specifics["batchsize"], shuffle=True, num_workers=2)
        self.optimizer=torch.optim.Adam(self.net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss_f=torch.nn.MSELoss(reduction='mean')
    def run(self,step):
        if step=="train":
            for epoch in range(self.specifics['niters']):
                for img, _ in iter(self.traindataloader):
                    img = img.cuda()
                    self.optimizer.zero_grad()
                    measurement = self.sensing(img)
                    img_hat = self.net(measurement)
                    loss = self.loss_f(img_hat, img)
                    loss.backward()
                    self.optimizer.step()
        if step=="test":
            with torch.no_grad():
                self.net.eval()
                val_psnrs = []
                for img, _ in iter(self.testdataloader):
                    img = img.cuda()
                    measurement = self.sensing(img)
                    img_hat = self.net(measurement)
                    val_psnrs.append(u.compute_average_psnr(img.cpu(), img_hat.detach().cpu()))
                val_psnr = sum(val_psnrs) / len(val_psnrs)
                print("average test psnr:"+str(val_psnr))
    # a function to return the reconstruction method with given parameters. It's a class with two methods: initialize and run.


