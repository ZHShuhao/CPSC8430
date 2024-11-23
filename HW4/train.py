import os
import torch
import torchvision
from torchvision import datasets, transforms
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser(description='GAN Trainer')
    parser.add_argument('--network', type=str, default='dcgan', help='GAN network')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    return parser.parse_args()


def train_epoch_dcgan(generator, discriminator, train_loader, criterion, optimizer_g, optimizer_d, device, epoch,
                      dim_noise):
    generator.train()
    discriminator.train()
    progress_bar = tqdm(train_loader)
    total_loss_d = 0
    total_loss_g = 0
    for i, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
        outputs = discriminator(images)
        loss_real = criterion(outputs, torch.ones_like(outputs).to(device))

        noise = torch.randn(images.size(0), dim_noise).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        loss_fake = criterion(outputs, torch.zeros_like(outputs).to(device))
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # 2. Update Generator: maximize log(D(G(z)))
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        loss_g = criterion(outputs, torch.ones_like(outputs).to(device))
        loss_g.backward()
        optimizer_g.step()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        progress_bar.set_description(f'Epoch: {epoch}. Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')

    total_loss_d /= len(train_loader)
    total_loss_g /= len(train_loader)
    return total_loss_d, total_loss_g


def train_epoch_wgan(generator, discriminator, train_loader, optimizer_g, optimizer_d, device, epoch, dim_noise):
    generator.train()
    discriminator.train()
    progress_bar = tqdm(train_loader)
    total_loss_d = 0
    total_loss_g = 0
    for i, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        # 1. Update Discriminator: maximize D(x) - D(G(z))
        for _ in range(5):
            optimizer_d.zero_grad()
            outputs = discriminator(images)
            loss_real = -torch.mean(outputs)

            noise = torch.randn(images.size(0), dim_noise).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            loss_fake = torch.mean(outputs)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

        # 2. Update Generator: maximize D(G(z))
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        loss_g = -torch.mean(outputs)
        loss_g.backward()
        optimizer_g.step()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        progress_bar.set_description(f'Epoch: {epoch}. Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')

    total_loss_d /= len(train_loader)
    total_loss_g /= len(train_loader)
    return total_loss_d, total_loss_g


def train_epoch_acgan(generator, discriminator, train_loader, criterion_gan, criterion_cls, optimizer_g, optimizer_d,
                      device, epoch, dim_noise):
    generator.train()
    discriminator.train()
    progress_bar = tqdm(train_loader)
    total_loss_d = 0
    total_loss_g = 0
    for i, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.long().to(device)

        # 1. Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()
        outputs, cls = discriminator(images)
        loss_real = criterion_gan(outputs, torch.ones_like(outputs).to(device))
        loss_cls_real = criterion_cls(cls, labels)

        noise = torch.randn(images.size(0), dim_noise).to(device)
        fake_labels = torch.randint(0, 10, (images.size(0),)).to(device)
        fake_images = generator(noise, fake_labels)
        outputs, cls = discriminator(fake_images.detach())
        loss_fake = criterion_gan(outputs, torch.zeros_like(outputs).to(device))
        loss_cls_fake = criterion_cls(cls, fake_labels)
        loss_d = loss_real + loss_fake + loss_cls_real + loss_cls_fake
        loss_d.backward()
        optimizer_d.step()

        # 2. Update Generator: maximize log(D(G(z)))
        optimizer_g.zero_grad()
        outputs, cls = discriminator(fake_images)
        loss_g = criterion_gan(outputs, torch.ones_like(outputs).to(device))
        loss_cls = criterion_cls(cls, fake_labels)
        loss_g = loss_g + loss_cls
        loss_g.backward()
        optimizer_g.step()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        progress_bar.set_description(f'Epoch: {epoch}. Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')

    total_loss_d /= len(train_loader)
    total_loss_g /= len(train_loader)
    return total_loss_d, total_loss_g


def test_epoch(generator, discriminator, device, epoch, dim_noise, network, fixed_noise=None):
    generator.eval()
    discriminator.eval()
    if fixed_noise is None:
        noise = torch.randn(25, dim_noise).to(device)
    else:
        noise = fixed_noise
    fake_images = generator(noise)

    save_path = f'./images/{network}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generated_images = fake_images.detach().cpu()
    generated_images = (generated_images + 1) / 2
    grid = torchvision.utils.make_grid(generated_images, nrow=5)
    torchvision.utils.save_image(grid, f'./images/{network}/epoch_{epoch + 1}.png')


def test_epoch_condition(generator, discriminator, device, epoch, dim_noise, network, fixed_noise, fixed_labels):
    generator.eval()
    discriminator.eval()
    if fixed_noise is None:
        noise = torch.randn(100, dim_noise).to(device)
    else:
        noise = fixed_noise
    if fixed_labels is None:
        labels = torch.arange(10).repeat(10).to(device)
    else:
        labels = fixed_labels
    fake_images = generator(noise, labels)

    save_path = f'./images/{network}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generated_images = fake_images.detach().cpu()
    generated_images = (generated_images + 1) / 2
    grid = torchvision.utils.make_grid(generated_images, nrow=10)
    torchvision.utils.save_image(grid, f'./images/{network}/epoch_{epoch + 1}.png')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dim_noise = 100
    dim_text = 100

    if args.network == 'dcgan':
        from net.dcgan import Generator, Discriminator
        generator = Generator(dim_noise).to(device)
        discriminator = Discriminator().to(device)

        criterion = torch.nn.BCELoss()
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        losses_d = []
        losses_g = []
        fixed_noise = torch.randn(25, dim_noise).to(device)
        for epoch in range(args.epochs):
            loss_d, loss_g = train_epoch_dcgan(generator, discriminator, train_loader,
                                               criterion, optimizer_g, optimizer_d,
                                               device, epoch, dim_noise)
            if (epoch + 1) % 5 == 0:
                test_epoch(generator, discriminator,
                           device, epoch, dim_noise,
                           args.network, fixed_noise)
            losses_d.append(loss_d)
            losses_g.append(loss_g)
    elif args.network == 'wgan':
        from net.wgan import Generator, Discriminator
        generator = Generator(dim_noise).to(device)
        discriminator = Discriminator().to(device)

        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
        optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

        losses_d = []
        losses_g = []
        fixed_noise = torch.randn(25, dim_noise).to(device)
        for epoch in range(args.epochs):
            loss_d, loss_g = train_epoch_wgan(generator, discriminator, train_loader,
                                              optimizer_g, optimizer_d,
                                              device, epoch, dim_noise)

            if (epoch + 1) % 5 == 0:
                test_epoch(generator, discriminator,
                           device, epoch, dim_noise,
                           args.network, fixed_noise)
            losses_d.append(loss_d)
            losses_g.append(loss_g)
    elif args.network == 'acgan':
        from net.acgan import Generator, Discriminator
        generator = Generator(dim_noise, dim_text).to(device)
        discriminator = Discriminator(len(classes)).to(device)

        criterion_gan = torch.nn.BCELoss()
        criterion_cls = torch.nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        losses_d = []
        losses_g = []
        fixed_noise = torch.randn(100, dim_noise).to(device)
        fixed_labels = torch.arange(10).repeat(10).to(device)
        for epoch in range(args.epochs):
            loss_d, loss_g = train_epoch_acgan(generator, discriminator, train_loader,
                                               criterion_gan, criterion_cls, optimizer_g, optimizer_d,
                                               device, epoch, dim_noise)
            if (epoch + 1) % 5 == 0:
                test_epoch_condition(generator, discriminator,
                                     device, epoch, dim_noise,
                                     args.network, fixed_noise, fixed_labels)
            losses_d.append(loss_d)
            losses_g.append(loss_g)

    torch.save(generator.state_dict(), f'./checkpoints/{args.network}_generator.pth')
    torch.save(discriminator.state_dict(), f'./checkpoints/{args.network}_discriminator.pth')

    # plot losses
    plt.plot(losses_d, label='Discriminator')
    plt.plot(losses_g, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{args.network}_loss.png')


if __name__ == '__main__':
    main()