import os
import time
import numpy as np
from skimage import io, img_as_float, transform as sk_transform
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error, normalized_mutual_information as mutual_info
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Define the dataset paths
data_dir = "/images/"
output_dir = "/denoised/"
os.makedirs(output_dir, exist_ok=True)

# Define the Denoising Autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 3, H, W)
            nn.Sigmoid()  # to ensure the output values are in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = DenoisingAutoencoder()

# Custom Dataset class for images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        print(f"Dataset initialized with {len(self.image_files)} samples.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = io.imread(img_name)
            image = img_as_float(image)
            image = sk_transform.resize(image, (256, 256), anti_aliasing=True)

            if self.transform:
                image = self.transform(image)

            return image, img_name  # Return the image and its filename
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            dummy_image = np.zeros((256, 256, 3), dtype=np.float32)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, img_name

# Paths and parameters
batch_size = 4
num_epochs = 50  # Increase the number of epochs for better training
learning_rate = 1e-3

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = ImageDataset(data_dir, transform=transform)
print(f"Number of samples in dataset: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training metrics
train_loss_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, _ in dataloader:
        inputs = inputs.float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)
    train_loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot the training loss
plt.plot(range(1, num_epochs+1), train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'denoising_autoencoder.pth')
print("Model saved!")

# Denoising and saving images
model.eval()
psnr_list = []
rmse_list = []
mutual_info_list = []
correlation_list = []
registration_times = []

for idx in range(len(dataset)):
    start_time = time.time()  # Record start time for registration

    image, img_name = dataset[idx]
    image = image.unsqueeze(0).float()  # Add batch dimension and ensure the tensor is of type float

    with torch.no_grad():
        denoised_image = model(image)

    denoised_image = denoised_image.squeeze(0).numpy().transpose(1, 2, 0)

    # Ensure denoised image is in the range [0, 1]
    denoised_image = np.clip(denoised_image, 0, 1)

    # Convert denoised image to uint8
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)

    # Save denoised image
    output_image_name = os.path.join(output_dir, os.path.basename(img_name))
    io.imsave(output_image_name, denoised_image_uint8)

    # Calculate metrics with images in range [0, 1]
    image_np = image.squeeze(0).numpy().transpose(1, 2, 0)  # Convert image to NumPy array
    psnr_value = psnr(image_np, denoised_image)
    rmse_value = np.sqrt(mean_squared_error(image_np, denoised_image))
    mutual_info_value = mutual_info(image_np.flatten(), denoised_image.flatten())
    correlation_value = pearsonr(image_np.flatten(), denoised_image.flatten())[0]

    # Check for invalid values
    if np.isnan(psnr_value) or np.isinf(psnr_value):
        psnr_value = 0
    if np.isnan(rmse_value) or np.isinf(rmse_value):
        rmse_value = 0
    if np.isnan(mutual_info_value) or np.isinf(mutual_info_value):
        mutual_info_value = 0
    if np.isnan(correlation_value) or np.isinf(correlation_value):
        correlation_value = 0

    psnr_list.append(psnr_value)
    rmse_list.append(rmse_value)
    mutual_info_list.append(mutual_info_value)
    correlation_list.append(correlation_value)

    # Record registration time
    end_time = time.time()
    registration_time = end_time - start_time
    registration_times.append(registration_time)

# Print average metrics and registration time
print(f'Average PSNR: {np.mean(psnr_list):.4f}')
print(f'Average RMSE: {np.mean(rmse_list):.4f}')
print(f'Average Mutual Information: {np.mean(mutual_info_list):.4f}')
print(f'Average Correlation Coefficient: {np.mean(correlation_list):.4f}')
print(f'Average Registration Time: {np.mean(registration_times):.4f} seconds')

print("Denoised images processed!")
