import torch
import torch.nn as nn


# Define the generator network for the ACGAN
class Generator(nn.Module):
    def __init__(self, dim_noise, dim_text, num_classes=10):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.dim_noise = dim_noise
        self.dim_text = dim_text
        self.text_embedding = nn.Embedding(self.num_classes, self.dim_text)
        self.linear = nn.Linear(self.dim_text + self.dim_noise, 4 * 4 * 512)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, noise, text):
        text = self.text_embedding(text)
        x = torch.cat([noise, text], dim=1)
        x = self.linear(x)
        x = x.view(-1, 512, 4, 4)
        return self.main(x)


# Define the discriminator network for the ACGAN
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear_gan = nn.Sequential(
            nn.Linear(4 * 4 * 512, 1),
            nn.Sigmoid()
        )
        self.linear_cls = nn.Sequential(
            nn.Linear(4 * 4 * 512, self.num_classes),
            nn.Softmax(dim=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image):
        x = self.main(image)
        x = x.view(-1, 512 * 4 * 4)
        return self.linear_gan(x), self.linear_cls(x)