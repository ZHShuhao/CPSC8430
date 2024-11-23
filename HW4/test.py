import os
import torch
import torchvision
from torchvision import datasets, transforms
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from scipy import linalg
import warnings
import matplotlib.pyplot as plt
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


def parse_args():
    parser = ArgumentParser(description='GAN Testing')
    parser.add_argument('--network', type=str, default='dcgan', help='GAN network')
    return parser.parse_args()


def evaluate(network, dataloader, netG, netD, device, dim_noise):
    iscore = InceptionScore().to(device)
    fid = FrechetInceptionDistance(feature=64).to(device)
    best_fid = float('inf')
    is_scores = []
    fid_scores = []
    for i, (real_image, label) in enumerate(dataloader):
        real_image = real_image.to(device)
        label = label.to(device)
        noise = torch.randn(real_image.size(0), dim_noise, device=device)
        if network == 'acgan':
            fake_image = netG(noise, label)
        else:
            fake_image = netG(noise)
        fake_image = (fake_image + 1) / 2

        real_images_ = torch.nn.functional.interpolate(real_image, size=(299, 299), mode='bilinear',
                                                       align_corners=False)
        real_images_ = (real_images_ * 255).to(torch.uint8)
        fake_images_ = torch.nn.functional.interpolate(fake_image, size=(299, 299), mode='bilinear',
                                                       align_corners=False)
        fake_images_ = (fake_images_ * 255).to(torch.uint8)

        is_score = iscore(fake_images_)
        is_scores.append(is_score[0].item())

        fid.update(real_images_, real=True)
        fid.update(fake_images_, real=False)
        fid_score = fid.compute()
        fid_scores.append(fid_score.item())
        fid.reset()

        if fid_score < best_fid:
            best_fid = fid_score
            torchvision.utils.save_image(fake_image, f'images/{network}/best_fake_image.png', nrow=5, normalize=True)

    is_scores = np.array(is_scores)
    fid_scores = np.array(fid_scores)
    print(f'Inception Score: {is_scores.mean()}')
    print(f'FID Score: {fid_scores.mean()}')


def load_model(network, dim_noise, dim_text, num_classes):
    if network == 'wgan':
        from net.wgan import Generator, Discriminator
        netG = Generator(dim_noise)
        netD = Discriminator()
    elif network == 'dcgan':
        from net.dcgan import Generator, Discriminator
        netG = Generator(dim_noise)
        netD = Discriminator()
    elif network == 'acgan':
        from net.acgan import Generator, Discriminator
        netG = Generator(dim_noise, dim_text)
        netD = Discriminator(num_classes)
    ckpt_g_path = os.path.join('checkpoints', f'{network}_generator.pth')
    ckpt_d_path = os.path.join('checkpoints', f'{network}_discriminator.pth')
    netG.load_state_dict(torch.load(ckpt_g_path))
    netD.load_state_dict(torch.load(ckpt_d_path))
    return netG, netD


def test(network, dim_noise, dim_text, num_classes, device):
    netG, netD = load_model(network, dim_noise, dim_text, num_classes)
    netG = netG.to(device).eval()
    netD = netD.to(device).eval()

    # load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    evaluate(network, dataloader, netG, netD, device, dim_noise)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_noise = 100
    dim_text = 100
    num_classes = 10
    test(args.network, dim_noise, dim_text, num_classes, device)


if __name__ == '__main__':
    main()