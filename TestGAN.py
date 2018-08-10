import math
import pickle
import torch
import torch.nn as nn
import numpy as np
import math, random
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
from scipy import ndimage as ndi
from skimage import feature

def PixelUpscale(LRimage):
    LRsize = LRimage.shape
    upLRimage = np.zeros([LRsize[0]*4, 4*LRsize[1], LRsize[2]])
    for i in range(4):
        for j in range(4):
            upLRimage[i::4,j::4,:] = LRimage[:,:,:]
    return(upLRimage)

def CompareImage(upLRimage,Bicubic,Outimage,HRTimage):
    HRsize = upLRimage.shape
    CompareI = np.zeros([HRsize[0], HRsize[1]*4, HRsize[2]])
    CompareI[:,:HRsize[1],:] = upLRimage[:,:,:]
    CompareI[:,HRsize[1]:HRsize[1]*2,:] = Bicubic[:,:,:]
    CompareI[:,HRsize[1]*2:HRsize[1]*3,:] = Outimage[:,:,:]
    CompareI[:,HRsize[1]*3:HRsize[1]*4,:] = HRTimage[:,:,:]
    return(CompareI)

def CompareImageDelimit(upLRimage,Bicubic,Outimage,HRTimage):
    HRsize = upLRimage.shape
    CompareI = np.ones([HRsize[0], HRsize[1]*4 + 30, HRsize[2]])
    CompareI[:,:HRsize[1],:] = upLRimage[:,:,:]
    CompareI[:,10+HRsize[1]:10+HRsize[1]*2,:] = Bicubic[:,:,:]
    CompareI[:,20+HRsize[1]*2:20+HRsize[1]*3,:] = Outimage[:,:,:]
    CompareI[:,30+HRsize[1]*3:30+HRsize[1]*4,:] = HRTimage[:,:,:]
    return(CompareI)

def BuildTestImage(LR, Generated, Target, epoch):
    upLR = PixelUpscale(LR.numpy().transpose((1,2,0)))
    hrTarget = Target.numpy().transpose((1,2,0))
    hrGenerated = PostprocessG(Generated)
    Bicubic = skimage.transform.rescale(LR.numpy().transpose((1,2,0)),4.0,order=3)
    TestImage = CompareImageDelimit(upLR ,Bicubic, hrGenerated, hrTarget)
    TestImage = Image.fromarray(np.uint8(TestImage*255))
    ZerosIncrement = ''
    if epoch < 10:
        ZerosIncrement += '0'
    if epoch < 100:
        ZerosIncrement += '0'
    if epoch < 1000:
        ZerosIncrement += '0'
    name = "./Results/GANtest/Epoch" + ZerosIncrement + str(epoch) + ".png"
    TestImage.save(name)

def ShowTestImage(LR, Generated, Target, epoch):
    upLR = PixelUpscale(LR.numpy().transpose((1,2,0)))
    hrTarget = Target.numpy().transpose((1,2,0))
    hrGenerated = PostprocessG(Generated)
    Bicubic = skimage.transform.rescale(LR.numpy().transpose((1,2,0)),4.0,order=3)
    TestImage = CompareImageDelimit(upLR ,Bicubic, hrGenerated, hrTarget)
    plt.imshow(TestImage)
    plt.show()

def PostprocessEdges(Edges):
    Edges = Edges.numpy().transpose((1,2,0))
    Edges3Channels = np.zeros([Edges.shape[0], Edges.shape[1], 3])
    Edges3Channels[:,:,0] = Edges[:,:,0]
    Edges3Channels[:,:,1] = Edges[:,:,0]
    Edges3Channels[:,:,2] = Edges[:,:,0]
    return Edges3Channels

def PostprocessG(Generated):
    HRTest= Generated.numpy().transpose((1,2,0))
    HRTest= (HRTest + np.ones_like(HRTest))*0.5
    HRTest[HRTest>1.0]=1.0
    HRTest[HRTest<0.0]=0.0
    return(HRTest)

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class opt(object):
    def __init__(self, batchSize = 10, nEpochs = 1002, lr=1e-4, step=500, threads = 0):
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.lr = lr
        self.step = step
        self.threads = threads

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(64)
        
    def forward(self, x):
        identity_data = x
        output = self.relu(self.bn1(self.conv1(self.pad1(x))))
        output = self.bn2(self.conv2(self.pad2(x)))
        output = torch.add(output,identity_data)
        return output

class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        npSobel = np.array([[[[1.0, 0.0, -1.0], [2., 0., -2], [1.0, 0.0, -1.0]]], [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],dtype=float)
        sobelTensor = torch.from_numpy(npSobel).float()
        self.SobelFilter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride =1, padding=1, bias= False)
        self.SobelFilter.weight.data = sobelTensor
    
    def forward(self, x):
        gray = torch.sum(x, dim=1)
        gray = gray.view(-1,1,256,256)
        GradImage = self.SobelFilter(gray)
        return GradImage

class FFTLoss(nn.Module):
    def __init__(self, batchsize):
        super(FFTLoss, self).__init__()
        self.eps = 1e-7
    
    def forward(self, x):
        x2 = Variable(x[:,:,16:-16,16:-16].data.clone(), requires_grad = True)
        x2 = x2.view(-1, 3*49, 32, 32)
        x = x.view(-1, 3*64, 32, 32)
        t = torch.cat((x, x2), 1)
        
       
        vF = torch.rfft(x, 2, onesided = False)
        vR = vF[:,:,:,0]
        vI = vF[:,:,:,1]
        
        out = torch.add(torch.pow(vR,2), torch.pow(vI,2))
        #out = torch.sqrt(out + self.eps)
        out = torch.log(out + 1)
        return out
    
class UpscaleNet(nn.Module):
    def __init__(self):
        super(UpscaleNet, self).__init__()
        
        self.pad_input = nn.ReflectionPad2d(4)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.pad_mid = nn.ReflectionPad2d(1)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64)

        self.upscale4x = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.pad_output = nn.ReflectionPad2d(4)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=0, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(self.pad_input(x)))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(self.pad_mid(out)))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(self.pad_output(out))
        return out

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64)
        )
        
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128)
        )
        
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128)
        )
        
        self.Block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256)
        )
        
        self.Block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
        )
        
        self.Block6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.Lrelu(self.conv_input(x))
        out = self.Block1(out)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.Block4(out)
        out = self.Block5(out)
        out = self.Block6(out)
        return out
        
def main(opt):
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(opt.lr)
    cudnn.benchmark = True
    
    print("===> Loading datasets")
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(256),
        transforms.ToTensor()])
    
    train_set = datasets.ImageFolder(root='./Texture/Train/',transform=data_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize*2, shuffle=False)
    test_set = datasets.ImageFolder(root='./Texture/Test/',transform=data_transform)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    print("===> Building model")
    UPmodel = UpscaleNet()
    D = DNet()
    UPmodel.load_state_dict(torch.load("./NetworkSaves/MSE_250.pt"))
    criterionBCE = nn.BCELoss().cuda()
    criterionMSE = nn.MSELoss(size_average=True).cuda()
    criterionL1 = nn.L1Loss(size_average=True).cuda()
    
    print("===> Setting GPU")
    UPmodel = UPmodel.cuda()
    D = D.cuda()
    
    UPmodel_parameters = filter(lambda p: p.requires_grad, UPmodel.parameters())
    params = sum([np.prod(p.size()) for p in UPmodel_parameters])
    print("Learnable parameters in UpscaleNet :", params)
    D_parameters = filter(lambda p: p.requires_grad, D.parameters())
    params = sum([np.prod(p.size()) for p in D_parameters])
    print("Learnable parameters in Discriminator :", params)
    
    print("===> Setting Optimizer")
    optimizer_Up = optim.Adam(UPmodel.parameters(), lr=opt.lr)
    optimizer_D = optim.Adam(D.parameters(), lr=0.2*opt.lr)
    scheduler_Up = lr_scheduler.StepLR(optimizer_Up, 70, gamma=0.1)
    scheduler_D = lr_scheduler.StepLR(optimizer_Up, 70, gamma=0.1)
    
    Sobel = SobelLoss().cuda()
    
    print("===> Training")
    EpochLossList = []
    G_LossList = []
    D_LossList = []
    MSE_LossList = []
    
    for epoch in range(0, opt.nEpochs + 1):
        a = trainNetwork(training_data_loader, testing_data_loader, optimizer_Up, optimizer_D, UPmodel, D, Sobel, criterionMSE, criterionBCE, criterionL1, epoch, opt.batchSize, EpochLossList, G_LossList, D_LossList, MSE_LossList)
        if a == 0:
            break
        scheduler_Up.step()
    
    np.savetxt('./Results/GANtest/Loss_GAN.txt', np.array(EpochLossList))
    np.savetxt('./Results/GANtest/G_GAN.txt', np.array(G_LossList))
    np.savetxt('./Results/GANtest/D_GAN.txt', np.array(D_LossList))
    np.savetxt('./Results/GANtest/MSE_GAN.txt', np.array(MSE_LossList))
    
    
    
    
def trainNetwork(training_data_loader, testing_data_loader, optimizer_Up, optimizer_D, UPmodel, D, Sobel, criterionMSE, criterionBCE, criterionL1, epoch, batch_size, EpochLossList, G_LossList, D_LossList, MSE_LossList):
    print ("epoch =", epoch)
    UPmodel.train()
    
    GLossList = []
    DLossList = []
    MSELossList = []
    LossList = []
    
    for iteration, batch in enumerate(training_data_loader, 1):
        if batch[0].size()[0] < 2*batch_size:
            break
        downSP = nn.AvgPool2d(4)
        LRinput = downSP(batch[0][:batch_size,:,:,:]).cuda()
        HRtarget = Variable(batch[0][:batch_size,:,:,:]*2-torch.ones(batch[0][:batch_size,:,:,:].size())).cuda()
        HRreal = Variable(batch[0][batch_size:,:,:,:]*2-torch.ones(batch[0][:batch_size,:,:,:].size())).cuda()
        HRFake = UPmodel(LRinput)
        
        #D Loss
        #Fake samples
        prediction_fake = D(HRFake.detach())
        d_loss_fake = criterionBCE(prediction_fake, Variable(torch.zeros(prediction_fake.size()).cuda()))
        
        #Real samples
        prediction_real = D(HRreal)
        d_loss_real = criterionBCE(prediction_real, Variable(torch.ones(prediction_real.size()).cuda()))
        
        #Combined
        d_loss = (d_loss_real + d_loss_fake)*0.5

        #Backprop and update
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        prediction_fake = D(HRFake)
        #G Loss
        MSELoss = criterionMSE(HRFake, HRtarget)   
        g_loss_GAN = criterionBCE(prediction_fake, Variable(torch.ones(prediction_fake.size()).cuda()))
        g_loss = MSELoss + 1e-4 * g_loss_GAN
        
        #Optimization   
        optimizer_Up.zero_grad()
        g_loss.backward()
        optimizer_Up.step()

        LossList.append(g_loss.data[0])
        MSELossList.append(MSELoss.data[0])
        G_LossList.append(g_loss_GAN.data[0])
        D_LossList.append(d_loss.data[0])
        
        #Plotting and following the progress
        if iteration%2000 == 0:
            print("===> Epoch[{}]({}/{}) Loss {:.5f}".format(epoch, iteration, len(training_data_loader), g_loss.data[0]))
            if epoch%1 == 0:
                UPmodel.eval()
                GTtest = iter(testing_data_loader).next()[0]
                LRtest = downSP(GTtest[:batch_size,:,:,:]).cuda()
                HRTest = UPmodel(LRtest)
                UPmodel.train()
                BuildTestImage(LRtest[0,:,:,:].data.cpu(), HRTest[0,:,:,:].data.cpu(), 
                               GTtest[0,:,:,:], epoch)
            if epoch%124 == 0:
                SRGname = "./NetworkSaves/GAN_Test_4x_" + str(epoch) + ".pt"
                torch.save(UPmodel.state_dict(),SRGname)
                
    LossMean = np.mean(np.array(LossList))
    MSEMean = np.mean(np.array(MSELossList))
    #GMean = np.mean(np.array(GLossList))
    #DMean = np.mean(np.array(DLossList))
    
    EpochLossList.append(LossMean)
    #G_LossList.append(GMean)
    #D_LossList.append(DMean)
    MSE_LossList.append(MSEMean)
    print("===> MSE {:.5f} Loss {:.5f}".format(MSEMean, LossMean))
    
print(torch.__version__)
options = opt(1,125)
main(options)