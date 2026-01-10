import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 64
batch_size = 64
nz = 100
ngf = 64
ndf = 64
nc = 3

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(
    root="Data/train",
    transform=transform
)

dog_label = 1
dog_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == dog_label]
dogs_only_dataset = Subset(full_dataset, dog_indices)
dogs_loader = DataLoader(dogs_only_dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
lr = 2e-4
beta1 = 0.5

optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

num_epochs = 50
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

os.makedirs("outputs/dcgan_dogs", exist_ok=True)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dogs_loader):
        real = images.to(device)
        b_size = real.size(0)

        D.zero_grad()
        real_labels = torch.ones(b_size, device=device)
        fake_labels = torch.zeros(b_size, device=device)

        output_real = D(real)
        lossD_real = criterion(output_real, real_labels)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = G(noise)
        output_fake = D(fake.detach())
        lossD_fake = criterion(output_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        G.zero_grad()
        output_fake_for_G = D(fake)
        lossG = criterion(output_fake_for_G, real_labels)
        lossG.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] LossD: {lossD.item():.4f} LossG: {lossG.item():.4f}")

    with torch.no_grad():
        fake_samples = G(fixed_noise).detach().cpu()
        fake_samples = (fake_samples + 1) / 2
        save_image(fake_samples,
                   f"outputs/dcgan_dogs/fake_epoch_{epoch+1:03d}.png",
                   nrow=8)

def generate_and_save_dogs(num_images=1000):
    os.makedirs("Data/train_variant_balanced/dogs", exist_ok=True)
    G.eval()
    count = 0
    with torch.no_grad():
        while count < num_images:
            cur_batch = min(64, num_images - count)
            noise = torch.randn(cur_batch, nz, 1, 1, device=device)
            fake = G(noise).detach().cpu()
            fake = (fake + 1) / 2
            for j in range(cur_batch):
                save_path = f"Data/train_variant_balanced/dogs/dog_fake_{count+j:05d}.png"
                save_image(fake[j], save_path)
            count += cur_batch
    print("Saved", count, "fake dog images.")

if __name__ == "__main__":
    pass
