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

###################################
##### Display functions ###########
###################################

def PixelUpscale(LRimage):
'''Funtion for displaying Nearest Neighbors interpolation in the result comparasion image'''
    LRsize = LRimage.shape
    upLRimage = np.zeros([LRsize[0]*4, 4*LRsize[1], LRsize[2]])
    for i in range(4):
        for j in range(4):
            upLRimage[i::4,j::4,:] = LRimage[:,:,:]
    return(upLRimage)

def CompareImage(upLRimage,Bicubic,Outimage,HRTimage):
    '''Generate a result image containing Ground Truth, NN interpolation, Bicubic Interpolation and Neural Network Super Resolution'''
    HRsize = upLRimage.shape
    CompareI = np.zeros([HRsize[0], HRsize[1]*4, HRsize[2]])
    CompareI[:,:HRsize[1],:] = upLRimage[:,:,:]
    CompareI[:,HRsize[1]:HRsize[1]*2,:] = Bicubic[:,:,:]
    CompareI[:,HRsize[1]*2:HRsize[1]*3,:] = Outimage[:,:,:]
    CompareI[:,HRsize[1]*3:HRsize[1]*4,:] = HRTimage[:,:,:]
    return(CompareI)

def CompareImageDelimit(upLRimage,Bicubic,Outimage,HRTimage):
'''Generate a result image containing Ground Truth, NN interpolation, Bicubic Interpolation and Neural Network Super Resolution
but with white separations between the images'''
    HRsize = upLRimage.shape
    CompareI = np.ones([HRsize[0], HRsize[1]*4 + 30, HRsize[2]])
    CompareI[:,:HRsize[1],:] = upLRimage[:,:,:]
    CompareI[:,10+HRsize[1]:10+HRsize[1]*2,:] = Bicubic[:,:,:]
    CompareI[:,20+HRsize[1]*2:20+HRsize[1]*3,:] = Outimage[:,:,:]
    CompareI[:,30+HRsize[1]*3:30+HRsize[1]*4,:] = HRTimage[:,:,:]
    return(CompareI)

def BuildTestImage(LR, Generated, Target, epoch):
'''Save and create the image using CompareImageDelimit to show training results
works for up to 9999 EpochLossList
Name and path of the file have to be input manually and cannot be specified as arguments'''
    #Nearest Neighbors using custom function
    #Transpose is needed because the image is saved with PIL image which indexes color channels on 1st component
    upLR = PixelUpscale(LR.numpy().transpose((1,2,0)))
    hrTarget = Target.numpy().transpose((1,2,0))
    #PostprocessG recenters the image (intially between -1 and 1), saturates values that are out of range for 
    #for display and reorders component
    hrGenerated = PostprocessG(Generated)
    #Bicubic using skimage
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
    name = "./Results/TextureCombx4GANfixed/Epoch" + ZerosIncrement + str(epoch) + ".png"
    TestImage.save(name)

def ShowTestImage(LR, Generated, Target, epoch):
'''Same as BuildTestImage but does not save the image and only displays the image if in a notebook'''
    upLR = PixelUpscale(LR.numpy().transpose((1,2,0)))
    hrTarget = Target.numpy().transpose((1,2,0))
    hrGenerated = PostprocessG(Generated)
    Bicubic = skimage.transform.rescale(LR.numpy().transpose((1,2,0)),4.0,order=3)
    TestImage = CompareImageDelimit(upLR ,Bicubic, hrGenerated, hrTarget)
    plt.imshow(TestImage)
    plt.show()

def PostprocessEdges(Edges):
'''Post processes the edges, not used in this iteration but can be used to show Edges or Sobel filtering in the comapre image
Basically just expands from 1 to 3 component to display with other color images'''
    Edges = Edges.numpy().transpose((1,2,0))
    Edges3Channels = np.zeros([Edges.shape[0], Edges.shape[1], 3])
    Edges3Channels[:,:,0] = Edges[:,:,0]
    Edges3Channels[:,:,1] = Edges[:,:,0]
    Edges3Channels[:,:,2] = Edges[:,:,0]
    return Edges3Channels

def PostprocessG(Generated):
'''Post processing of the Neural Net image
Center from [-1, 1] (Target Image is centered, Generator ends with a Convolutional layer) to [0,1], saturates for dispaly reasons and from pytorch tensor to numpy array'''
    HRTest= Generated.numpy().transpose((1,2,0))
    HRTest= (HRTest + np.ones_like(HRTest))*0.5
    HRTest[HRTest>1.0]=1.0
    HRTest[HRTest<0.0]=0.0
    return(HRTest)

def rgb2gray(rgb):
'''Classic rgb to gray function'''
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

###################################
##### Neural Network Modules ######
###################################

class opt(object):
'''Class containing options for training the network
step is useless as it has been replaced by pytorch scheduler'''
    def __init__(self, batchSize = 10, nEpochs = 1002, lr=1e-4, step=500, threads = 0):
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.lr = lr
        self.step = step
        self.threads = threads

class _Residual_Block(nn.Module):
'''Definition of Residual Blocks
Reflection Padding for all convs
64 channels
3 kernel size (stride 1)
Instance Normalisation
Leaky ReLU 0.2
'''
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
        #add output of the ResBlock with its input with a skip connection
        output = torch.add(output,identity_data)
        return output

class SobelLoss(nn.Module):
'''Computation of the 1st order derivative of the image with a Sobel Filter to emphasize edges'''
    def __init__(self):
        super(SobelLoss, self).__init__()
        #definition of the Sobel filter
        npSobel = np.array([[[[1.0, 0.0, -1.0], [2., 0., -2], [1.0, 0.0, -1.0]]], [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],dtype=float)
        sobelTensor = torch.from_numpy(npSobel).float()
        #Convolution with fixed weights
        #Outputs the Horizontal and Vertical derivatives as 2 different channels (they will be combined later in the loss function)
        self.SobelFilter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride =1, padding=1, bias= False)
        self.SobelFilter.weight.data = sobelTensor
    
    def forward(self, x):
        #Only apply the Sobel filter to the grayscale image (no weighting by color)
        gray = torch.sum(x, dim=1)
        gray = gray.view(-1,1,256,256)
        GradImage = self.SobelFilter(gray)
        return GradImage

class FFTLoss(nn.Module):
'''Compute the Fourier spectrum of the image to compare frequencies
It is done by patch so the information is not totally global information
In this implementation the windows are of size 32x32 (out of a 256x256 image or RandomCrop)'''
    def __init__(self, batchsize):
        super(FFTLoss, self).__init__()
        self.eps = 1e-7
    
    def forward(self, x):
        x2 = Variable(x[:,:,16:-16,16:-16].data.clone(), requires_grad = True)
        x2 = x2.view(-1, 3*49, 32, 32)
        x = x.view(-1, 3*64, 32, 32)
        t = torch.cat((x, x2), 1)
        
       #Extract the FFT using torch algorithm
        vF = torch.rfft(x, 2, onesided = False)
        #get real part
        vR = vF[:,:,:,0]
        #get the imaginary part
        vI = vF[:,:,:,1]
        
        #Get spectrum by computing the elemcent wise complex modulus
        out = torch.add(torch.pow(vR,2), torch.pow(vI,2))
        out = torch.sqrt(out + self.eps)
        return out
    
class UpscaleNet(nn.Module):
'''Definition of the Super Resolution Network (Generator in Adversarial training)
All padding are Reflect Padding,
Using Instance Normalisation
16 ResBlocks (ch=64, k=3, LRelu, Instance Norm)
Skip connection
Upscaling with PixelShuffle (Fractional Convolution)
'''
    def __init__(self):
        super(UpscaleNet, self).__init__()
        #Input is 3 Channels => to 64 channels (kernel size 9 stride 1)
        self.pad_input = nn.ReflectionPad2d(4)
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        #Running 16 resBlocks        
        self.residual = self.make_layer(_Residual_Block, 16)
        
        # Position of the Network wise skip connection(transfering spatial information from the low dimension image)
        self.pad_mid = nn.ReflectionPad2d(1)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64)

        #Upscale module with PixelShuffle (Fractional Convolution)
        self.upscale4x = nn.Sequential(
            #using x2 two times to upscale by 4 times
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        #Final convolution to aleviate artifacts from the upsampling and colapsing channels from 64 to 3
        self.pad_output = nn.ReflectionPad2d(4)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=0, bias=False)
    
        for m in self.modules():
        #Initialisation of conv layers
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
    #Generate the ResBlocks
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
    #FeedForward
        out = self.relu(self.conv_input(self.pad_input(x)))
        #save input conv for skip connection
        residual = out
        #Apply ResBlocks
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(self.pad_mid(out)))
        #Apply skip connection
        out = torch.add(out,residual)
        #Upsclae
        out = self.upscale4x(out)
        #Generate output
        out = self.conv_output(self.pad_output(out))
        return out

class DNet(nn.Module):
'''Discriminator Network as a Encoder
See Poster or Abstract for easy and clear view of the architecture'''
    def __init__(self):
        super(DNet, self).__init__()
        
        #Input from 3 to 64 channels
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.Block0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64)
        )
        
        #Reduce the size of the image (stride = 2)
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64)
        )
        
        #Augment the number of channels
        self.Block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128)
        )
        
        #Reduce the size of the image (stride = 2)
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128)
        )
        
        #Augment the number of channels
        self.Block4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(256)
        )
        
        #Augment the number of channels
        #Reduce the size of the image (stride = 2)
        self.Block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        #Output is not a scalar but a small 2D matrix (1 channel), the Discriminator is a Patch GAN
        #Patch GAN is very good in SR because the total coherence of the image is almost never lost but the small details are.
        #We aim at a local reconstruction with context which is what a PatchGAN with a large receptive field can does
        #Additionally it makes it so no fully connected layer is needed which makes the Network less memory heavy
        self.Block6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
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
    #FeedForward
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
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize*3, shuffle=True)
    test_set = datasets.ImageFolder(root='./Texture/Test/',transform=data_transform)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    
    print("===> Building model")
    UPmodel = UpscaleNet()
    D = DNet()
    UPmodel.load_state_dict(torch.load("./NetworkSaves/MSE_250.pt"))
    criterionBCE = nn.BCELoss(size_average=True).cuda()
    criterionMSE = nn.MSELoss(size_average=True).cuda()
    criterionL1 = nn.L1Loss(size_average=True).cuda()
    
    print("===> Setting GPU")
    UPmodel = UPmodel.cuda()
    D = D.cuda()
    UPmodel_parameters = filter(lambda p: p.requires_grad, UPmodel.parameters())

    UPmodel_parameters = filter(lambda p: p.requires_grad, UPmodel.parameters())
    params = sum([np.prod(p.size()) for p in UPmodel_parameters])
    print("Learnable parameters in UpscaleNet :", params)
    D_parameters = filter(lambda p: p.requires_grad, D.parameters())
    params = sum([np.prod(p.size()) for p in D_parameters])
    print("Learnable parameters in Discriminator :", params)
    
    print("===> Setting Optimizer")
    optimizer_Up = optim.Adam(UPmodel.parameters(), lr=opt.lr)
    optimizer_D = optim.Adam(D.parameters(), lr=opt.lr)
    scheduler_Up = lr_scheduler.StepLR(optimizer_Up, 225, gamma=0.1)
    scheduler_D = lr_scheduler.StepLR(optimizer_Up, 225, gamma=0.1)
    
    Sobel = SobelLoss().cuda()
    
    print("===> Training")
    EpochLossList = []
    G_LossList = []
    D_LossList = []
    MSE_LossList = []
    S_LossList = []
    FFT_LossList =[]
    
    for epoch in range(0, opt.nEpochs + 1):
        a = trainNetwork(training_data_loader, testing_data_loader, optimizer_Up, optimizer_D, UPmodel, D, Sobel, criterionMSE, criterionBCE, criterionL1, epoch, opt.batchSize, EpochLossList, G_LossList, D_LossList, MSE_LossList, S_LossList, FFT_LossList)
        if a == 0:
            break
        scheduler_Up.step()
        scheduler_D.step()
    
    np.savetxt('./Results/TextureCombx4GAN/Loss/Loss_GAN.txt', np.array(EpochLossList))
    np.savetxt('./Results/TextureCombx4GAN/Loss/G_GAN.txt', np.array(G_LossList))
    np.savetxt('./Results/TextureCombx4GAN/Loss/D_GAN.txt', np.array(D_LossList))
    np.savetxt('./Results/TextureCombx4GAN/Loss/MSE_GAN.txt', np.array(MSE_LossList))
    np.savetxt('./Results/TextureCombx4GAN/Loss/S_GAN.txt', np.array(S_LossList))
    np.savetxt('./Results/TextureCombx4GAN/Loss/FFT_GAN.txt', np.array(FFT_LossList))
    
    
def trainNetwork(training_data_loader, testing_data_loader, optimizer_Up, optimizer_D, UPmodel, D, Sobel, criterionMSE, criterionBCE, criterionL1, epoch, batch_size, EpochLossList, G_LossList, D_LossList, MSE_LossList, S_LossList, FFT_LossList):
    print ("epoch =", epoch)
    UPmodel.train()
    
    GLossList = []
    DLossList = []
    MSELossList = []
    LossList = []
    SLossList = []
    FFTLossList = []
    
    for iteration, batch in enumerate(training_data_loader, 1):
        if batch[0].size()[0] < 3*batch_size:
            break
        
        ones = torch.ones(batch[0][:batch_size,:,:,:].size())
        #1st sampling for Discriminator
        downSP = nn.AvgPool2d(4)
        LRinput = downSP(batch[0][:batch_size,:,:,:]).cuda()
        HRreal = Variable(batch[0][batch_size:batch_size*2,:,:,:]*2-ones).cuda()
        HRFake = UPmodel(LRinput)
        
        #D Loss
        #Precompute Doutput samples
        prediction_fake = D(HRFake.detach())
        prediction_real = D(HRreal)
                                  
        #Logits
        logits0 = Variable(torch.ones(prediction_fake.size()).cuda(), requires_grad = False)
        logits1 = Variable(torch.zeros(prediction_fake.size()).cuda(), requires_grad = False)
        
        #Fake samples
        d_loss_fake = criterionBCE(prediction_fake, logits0)
        
        #Real samples  
        d_loss_real = criterionBCE(prediction_real, logits1)
        
        #Combined
        d_loss = (d_loss_real + d_loss_fake)*0.5

        #Backprop and update
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        del LRinput, HRreal, HRFake, prediction_fake, prediction_real
        
        #Second Sampling for Generator
        LRinput = downSP(batch[0][batch_size*2:batch_size*3,:,:,:]).cuda()
        HRtarget = Variable(batch[0][batch_size*2:batch_size*3,:,:,:]*2-ones).cuda()
        HRFake = UPmodel(LRinput)
        
        prediction_fake = D(HRFake)
        
        #MSE Loss
        MSELoss = criterionMSE(HRFake, HRtarget)
        
        #Edge Loss
        SLoss = criterionMSE(Sobel(HRFake), Sobel(HRtarget).detach())
        
        #Texture Loss
        FFTLossTarget = FFTLoss(batch_size)
        FFTLossGene = FFTLoss(batch_size)
        MT = FFTLossTarget(HRtarget).detach()
        MG = FFTLossGene(HRFake)
        FFTLossL1 = criterionL1(MG, MT)
        
        #GAN Loss
        g_loss_GAN = criterionBCE(prediction_fake, logits1)
        
        #Hyperparameters
        alphaGAN = 1e-2
        alphaS = 1e-2
        alphaFFT = 1e-3
        
        #Total Loss
        g_loss = MSELoss + alphaGAN * g_loss_GAN + alphaS*SLoss + alphaFFT*FFTLossL1
        
        #Optimization   
        optimizer_Up.zero_grad()
        g_loss.backward()
        optimizer_Up.step()
                          
        del LRinput, HRtarget, HRFake, prediction_fake

        LossList.append(g_loss.data[0])
        MSELossList.append(MSELoss.data[0])
        GLossList.append(g_loss_GAN.data[0])
        DLossList.append(d_loss.data[0])
        SLossList.append(SLoss.data[0])
        FFTLossList.append(FFTLossL1.data[0])
        
        #Plotting and following the progress
        if iteration%140 == 0:
            print("===> Epoch[{}]({}/{}) Loss {:.5f}".format(epoch, iteration, len(training_data_loader), g_loss.data[0]))
            if epoch%1 == 0:
                UPmodel.eval()
                GTtest = iter(testing_data_loader).next()[0]
                LRtest = downSP(GTtest[:batch_size,:,:,:]).cuda()
                HRTest = UPmodel(LRtest)
                UPmodel.train()
                BuildTestImage(LRtest[0,:,:,:].data.cpu(), HRTest[0,:,:,:].data.cpu(), 
                               GTtest[0,:,:,:], epoch)
            if epoch%275 == 0:
                SRGname = "./NetworkSaves/GAN_Comb_4x_fixed" + str(epoch) + ".pt"
                torch.save(UPmodel.state_dict(),SRGname)
                
    LossMean = np.mean(np.array(LossList))
    MSEMean = np.mean(np.array(MSELossList))
    GMean = np.mean(np.array(GLossList))
    DMean = np.mean(np.array(DLossList))
    SMean = np.mean(np.array(SLossList))
    FFTMean = np.mean(np.array(FFTLossList))
    
    EpochLossList.append(LossMean)
    G_LossList.append(GMean)
    D_LossList.append(DMean)
    MSE_LossList.append(MSEMean)
    print("===> MSE {:.5f} G {:.5f} D {:.5f} Sobel {:.5f} FFT {:.5f} Loss {:.5f}".format(MSEMean, GMean, DMean, SMean, FFTMean, LossMean))
    
print(torch.__version__)
options = opt(6,551)
main(options)