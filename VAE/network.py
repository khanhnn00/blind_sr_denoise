import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 64, 6, 6)

class VAE(nn.Module):
    def __init__(self, image_channels=3, z_dim=32):
        super(VAE, self).__init__()

        h_dim = 2304

        # (batch_size, ch, h, w)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),    # torch.Size([32, 2304, 1, 1])
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=3, stride=2),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print('h.shape', h.shape)     # h.shape torch.Size([32, 2304])
        z, mu, logvar = self.bottleneck(h)
        # print('z.shape', z.shape)     # z.shape torch.Size([32, 16])
        return z, mu, logvar

    def decode(self, z):
        a = self.fc3(z)
        # print('a.shape', a.shape)  # a.shape torch.Size([32, 2304])
        x = self.decoder(a)
        # print('x.shape', x.shape)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar