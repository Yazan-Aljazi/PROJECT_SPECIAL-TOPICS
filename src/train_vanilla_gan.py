import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


dogs_dir = "../Data/train/dogs"  

img_size = 64
batch_size = 64

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

dataset = datasets.ImageFolder(
    root="Data/train",  
    transform=transform,
)

print("Classes:", dataset.classes)
dog_label_index = dataset.classes.index("dogs")

dog_indices = [i for i, (_, label) in enumerate(dataset) if label == dog_label_index]
dog_subset = torch.utils.data.Subset(dataset, dog_indices)

dataloader = DataLoader(dog_subset, batch_size=batch_size, shuffle=True)

print("Number of dog images used for GAN training:", len(dog_subset))

z_dim = 100  

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels=3, feature_g=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (N, 3, 64, 64)
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability real/fake
        )

    def forward(self, x):
        return self.net(x).view(-1)


G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
lr = 2e-4

optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

epochs = 120 

os.makedirs("outputs/vanilla_gan", exist_ok=True)

fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        # Traning the Discriminator 
        D.zero_grad()

        # real data (label=1)
        real_labels = torch.ones(batch_size_curr, device=device)
        output_real = D(real_imgs)
        loss_D_real = criterion(output_real, real_labels)

        # fake data G (label=0)
        noise = torch.randn(batch_size_curr, z_dim, 1, 1, device=device)
        fake_imgs = G(noise)
        fake_labels = torch.zeros(batch_size_curr, device=device)
        output_fake = D(fake_imgs.detach())
        loss_D_fake = criterion(output_fake, fake_labels)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        
        G.zero_grad()
      
        output_fake_for_G = D(fake_imgs)
        loss_G = criterion(output_fake_for_G, real_labels)
        loss_G.backward()
        optimizer_G.step()

        if i % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}"
            )

   
    with torch.no_grad():
        fake_samples = G(fixed_noise).detach().cpu()
      
        fake_samples = (fake_samples + 1) / 2
        utils.save_image(
            fake_samples,
            f"outputs/vanilla_gan/fake_epoch_{epoch+1}.png",
            nrow=8,
        )
torch.save(G.state_dict(), "outputs/vanilla_gan/generator_vanilla.pth")
torch.save(D.state_dict(), "outputs/vanilla_gan/discriminator_vanilla.pth")

print("Training finished and models saved.")
