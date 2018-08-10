import pickle
import torch
import torch.nn as nn
import numpy as np
import math, random
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
from scipy import ndimage as ndi
from skimage import feature, measure
import gc

def PixelUpscale(LRimage):
    LRsize = LRimage.shape
    upLRimage = np.zeros([LRsize[0]*8, 8*LRsize[1], LRsize[2]])
    for i in range(8):
        for j in range(8):
            upLRimage[i::8,j::8,:] = LRimage[:,:,:]
    return(upLRimage)

def CompareImage(upLRimage,Bicubic,Outimage,HRTimage):
    HRsize = upLRimage.shape
    CompareI = np.zeros([HRsize[0], HRsize[1]*4, HRsize[2]])
    CompareI[:,:HRsize[1],:] = upLRimage[:,:,:]
    CompareI[:,HRsize[1]:HRsize[1]*2,:] = Bicubic[:,:,:]
    CompareI[:,HRsize[1]*2:HRsize[1]*3,:] = Outimage[:,:,:]
    CompareI[:,HRsize[1]*3:HRsize[1]*4,:] = HRTimage[:,:,:]
    return(CompareI)

def CompareImageDelimit(Outimage):
    HRsize = Outimage.shape
    CompareI = np.ones([HRsize[0], HRsize[1], HRsize[2]])
    CompareI[:,:,:] = Outimage[:,:,:]
    return(CompareI)

def BuildTestImage(LR, epoch):
    upLR = PixelUpscale(LR.data.cpu().numpy().transpose((1,2,0)))
    TestImage = CompareImageDelimit(skimage.transform.rescale(LR.cpu().numpy().transpose((1,2,0)),4.0,order=3))
    TestImage = Image.fromarray(np.uint8(TestImage*255))
    ZerosIncrement = ''
    if epoch < 10:
        ZerosIncrement += '0'
    if epoch < 100:
        ZerosIncrement += '0'
    if epoch < 1000:
        ZerosIncrement += '0'
    name = "./NetworkEvaluation/Bicubic/" + ZerosIncrement + str(epoch) + ".png"
    TestImage.save(name)

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
    def __init__(self, batchSize = 1, nEpochs = 1002, lr=1e-4, step=500, threads = 1):
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

class UpscaleNet(nn.Module):
    def __init__(self):
        super(UpscaleNet, self).__init__()
        
        self.pad_input = nn.ReflectionPad2d(4)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 20)

        self.pad_mid = nn.ReflectionPad2d(1)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64)

        self.upscale8x = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64)
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
        out = self.upscale8x(out)
        out = self.conv_output(self.pad_output(out))
        return out
    
class LossStruct(object):
    def __init__(self):
        self.MSELoss = []
        self.GBCELoss = []
        self.DLoss = []
        self.DLossFake = []
        self.DLossReal = []
        self.GLoss = []
        self.AdvEpoch =None
    
    def updateDloss(self,valueReal, valueFake, value):
        if not self.DLoss:
            if not self.MSELoss:
                self.AdvEpoch = 0
            else:
                self.AdvEpoch = len(self.MSELoss)
        self.DLossReal.append(valueReal)
        self.DLossFake.append(valueFake)
        self.DLoss.append(value)
    
    def updateGLoss(self,valueMSE,valueBCE,valueG):
        self.MSELoss.append(valueMSE)
        self.GBCELoss.append(valueBCE)
        self.GLoss.append(valueG)
        
    def updateResNet(self,valueMSE,valueG):
        self.MSELoss.append(valueMSE)
        self.GLoss.append(valueG)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.UpscaleNet = UpscaleNet()
        self.EdgesNet = EdgesNet()
        
    def forward(self,x):
        return(self.UpscaleNet(Variable(x)))

def main(opt):
    if not torch.cuda.is_available():
        print('no cuda')
        raise Exception("No GPU avialable")

    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(opt.lr)
    cudnn.benchmark = True
    
    print("===> Loading datasets")
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    
    #train_set = datasets.ImageFolder(root='./yosemiteImages/train',transform=transforms.ToTensor())
    #training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    test_set = datasets.ImageFolder(root='./Texture/Test/',transform=transforms.ToTensor())
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    
    
    print("===> Building model")
    model = UpscaleNet().cuda()
    
    model.eval()
    
    print("===> Forward")
    for iteration, batch in enumerate(testing_data_loader, 1):
        print(iteration)
        downSP = nn.AvgPool2d(4)
        batchM = batch[0][:,:,:,:]
        LRinput = Variable(downSP(batch[0])).cuda()
        #Target = PostprocessG(HRtarget[0,:,:,:].data.cpu())
        #Save
        BuildTestImage(LRinput[0], iteration)
    
    #np.savetxt('./NetworkEvaluation/SSIM_test_FFT3net.txt',np.array(SSIM))
    #np.savetxt('./NetworkEvaluation/PSNR_test_FFT3net.txt',np.array(PSNR))
options = opt()
main(options)
print('stop')
