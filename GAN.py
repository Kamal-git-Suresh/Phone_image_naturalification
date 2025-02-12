import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9):
        super(Generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)



class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**3, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



def get_loss(device):
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_L1 = torch.nn.L1Loss().to(device)
    criterion_Perceptual = PerceptualLoss().to(device)

    return criterion_GAN, criterion_L1, criterion_Perceptual

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19().features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        return self.criterion(x_vgg, y_vgg.detach())



class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

class JPEGRawDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.jpeg_dir = os.path.join(root_dir, "jpeg")
        self.raw_dir = os.path.join(root_dir, "raw")
        self.image_names = [f for f in os.listdir(self.jpeg_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        jpeg_path = os.path.join(self.jpeg_dir, img_name)
        raw_path = os.path.join(self.raw_dir, img_name.replace(".jpg", ".png"))

        jpeg_image = Image.open(jpeg_path).convert("RGB")
        raw_image = Image.open(raw_path).convert("RGB")


        if self.transform:
            jpeg_image = self.transform(jpeg_image)
            raw_image = self.transform(raw_image)


        return jpeg_image, raw_image


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 100

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    init_weights(netG, init_type='kaiming')
    init_weights(netD, init_type='kaiming')

    criterion_GAN, criterion_L1, criterion_Perceptual = get_loss(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = JPEGRawDataset(root_dir="path/to/your/dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    for epoch in range(num_epochs):
        for i, (jpeg_images, raw_images) in enumerate(dataloader):
            jpeg_images = jpeg_images.to(device)
            raw_images = raw_images.to(device)

            optimizerD.zero_grad()

            real_labels = torch.ones(jpeg_images.size(0), device=device)
            output_real = netD(raw_images).view(-1)
            errD_real = criterion_GAN(output_real, real_labels)
            errD_real.backward()
            D_x = output_real.mean().item()

            fake_images = netG(jpeg_images)
            fake_labels = torch.zeros(jpeg_images.size(0), device=device)
            output_fake = netD(fake_images.detach()).view(-1)
            errD_fake = criterion_GAN(output_fake, fake_labels)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            optimizerG.zero_grad()
            real_labels = torch.ones(jpeg_images.size(0), device=device)
            output = netD(fake_images).view(-1)
            errG_GAN = criterion_GAN(output, real_labels)


            errG_L1 = criterion_L1(fake_images, raw_images)

            errG_Perceptual = criterion_Perceptual(fake_images, raw_images)

            errG = errG_GAN + 10*errG_L1 + 0.5*errG_Perceptual

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()



            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")


        if epoch % 10 == 0:
            torch.save(netG.state_dict(), f"generator_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"discriminator_epoch_{epoch}.pth")