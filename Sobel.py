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
    name = "./Results/Sobel_1e-1/Epoch" + ZerosIncrement + str(epoch) + ".png"
    TestImage.save(name)

def SaveBatch(Batch, text):
    TestImage = np.zeros([256,len(Batch)*256,3])
    for i in range(len(Batch)):
        TestImage[:,i*256:(i+1)*256,:] = Batch[i].numpy().transpose((1,2,0))
        
    TestImage = (TestImage + np.ones_like(TestImage))*0.5
    TestImage = Image.fromarray(np.uint8(TestImage*255))
    ZerosIncrement = ''
    name = "./Results/FFT2k/" + ZerosIncrement + text + ".png"
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
    def __init__(self, batchSize = 10, nEpochs = 1002, lr=1e-4, step=500, threads = 8):
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
        sobelTensor = torch.tensor([[[[1.0, 0.0, -1.0], [2., 0., -2], [1.0, 0.0, -1.0]]], [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]])
        self.SobelFilter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride =1, padding=1, bias= False)
        self.SobelFilter.weight.data = sobelTensor
    
    def forward(self, x):
        gray = torch.sum(x, dim=1)
        gray = gray.view(-1,1,256,256)
        GradImage = self.SobelFilter(gray)
        return GradImage

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
        
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.UpscaleNet = UpscaleNet()
        self.EdgesNet = EdgesNet()
        
    def forward(self,x):
        return(self.UpscaleNet(Variable(x)))

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
    
    train_set = datasets.ImageFolder(root='./Div2K/Train/',transform=data_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_set = datasets.ImageFolder(root='./Div2K/Validation/',transform=data_transform)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    print("===> Building model")
    model = UpscaleNet()
    #model.load_state_dict(torch.load("./NetworkSaves/INResnet1000.pt"))
    criterionMSE = nn.MSELoss(size_average=True).cuda()
    criterionL1 = nn.L1Loss(size_average=True).cuda()
    
    print("===> Setting GPU")
    model = model.cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Learnable parameters in UpscaleNet :", params)
    
    print("===> Setting Optimizer")
    optimizer_Up = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler_Up = lr_scheduler.StepLR(optimizer_Up, 400, gamma=0.1)
    
    print("===> Setting Optimizer")
    Sobel = SobelLoss().cuda()
    
    print("===> Training")
    EpochLossList = []
    for epoch in range(0, opt.nEpochs + 1):
        a = trainNetwork(training_data_loader, testing_data_loader, optimizer_Up, model, Sobel, criterionMSE, 
                     criterionL1, epoch, opt.batchSize, EpochLossList)
        if a == 0:
            break
        scheduler_Up.step()
    
    np.savetxt('./Results/Sobel2k.txt', np.array(EpochLossList))
    
    
def trainNetwork(training_data_loader, testing_data_loader, optimizer_Up, model, Sobel, criterionMSE, criterionL1, epoch, batch_size, EpochLossList):
    print ("epoch =", epoch)
    model.train()
    LossList = []
    SLossList = []
    MSELossList = []

    
    for iteration, batch in enumerate(training_data_loader, 1):
        LRinput = Variable(0.25*batch[0][:batch_size,:,::4,::4]+0.25*batch[0][:batch_size,:,1::4,1::4]+
                           0.25*batch[0][:batch_size,:,2::4,2::4]+0.25*batch[0][:batch_size,:,3::4,3::4]).cuda()
        HRtarget = Variable(batch[0][:batch_size,:,:,:]*2-torch.ones(batch[0][:batch_size,:,:,:].size())).cuda()
        GeneratedImage = model(LRinput)
        
        HRtargetNP = HRtarget.data.cpu().numpy().ravel()
        #Content Loss
        MSELoss = criterionMSE(GeneratedImage, HRtarget)
        SLoss = criterionMSE(Sobel(GeneratedImage), Sobel(HRtarget).detach())
        alpha = 1e-1
        Loss = MSELoss + alpha*SLoss
        
        minT = np.min(HRtargetNP)
        maxT = np.max(HRtargetNP)
        
        meanT = np.mean(HRtargetNP)
        
        MeanRe = np.mean(np.reshape(np.sum(HRtarget.data.cpu().numpy()[0]/3.0, axis = 0 ),
                   [16, 64*64]), axis=1 )
        Bool = np.abs(np.min(MeanRe) - maxT) < 0.05 or np.abs(np.max(MeanRe) - minT) < 0.05

        
        if meanT > 0.999 or meanT < -0.999 or Bool:
            pass
        else:        
            #Optimization
            optimizer_Up.zero_grad()
            Loss.backward()
            optimizer_Up.step()

        LossList.append(Loss.data[0])
        MSELossList.append(MSELoss.data[0])
        SLossList.append(alpha*SLoss.data[0])
        
        #Plotting and following the progress
        if iteration%800 == 0:
            print("===> Epoch[{}]({}/{}) MSE {:.5f}, SLoss {:.5f}, Loss {:.5f}".format(epoch, iteration, len(training_data_loader), MSELoss.data[0], alpha*SLoss.data[0], Loss.data[0]))
            if epoch%1 == 0:
                model.eval()
                GTtest = iter(testing_data_loader).next()[0]
                LRtest = Variable(0.25*GTtest[:batch_size,:,::4,::4]+0.25*GTtest[:batch_size,:,1::4,1::4]+
                           0.25*GTtest[:batch_size,:,2::4,2::4]+0.25*GTtest[:batch_size,:,3::4,3::4]).cuda()
                HRTest = model(LRtest)
                model.train()
                BuildTestImage(LRtest[0,:,:,:].data.cpu(), HRTest[0,:,:,:].data.cpu(), 
                               GTtest[0,:,:,:], epoch)
            if epoch%500 == 0:
                SRGname = "./NetworkSaves/Sobel2k" + str(epoch) + ".pt"
                torch.save(model.state_dict(),SRGname)
                
    LossMean = np.mean(np.array(LossList))
    MSELossMean = np.mean(np.array(MSELossList))
    SLossMean = np.mean(np.array(SLossList))
    EpochLossList.append(LossMean)
    print("===> MSE {:.5f}, SLoss {:.5f}, Loss {:.5f}".format(MSELossMean, SLossMean, LossMean))
    
print(torch.__version__)
options = opt(1,1002)
main(options)
