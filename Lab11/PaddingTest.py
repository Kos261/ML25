import torch
import torch.nn as nn

# Step 1: Define the input tensor (NCHW format)
# x = torch.tensor([[[[2., 3., 1.],
#                     [0., 4., 2.],
#                     [1., 6., 7.]]]])  # shape: [1, 1, 3, 3]

x = torch.tensor([[[[2., 3.],
                    [0., 4.],]]])  # shape: [1, 1, 2, 2]

# Step 2: Define the transposed convolution layer
convT = nn.ConvTranspose2d(
    in_channels=1, 
    out_channels=1,
    kernel_size=2,
    stride=2,
    padding=0,
    output_padding=1,
    bias=False
)
#Step 3: manually define kernel
kernel = torch.tensor([[[[1., 2.],
                          [3., 4.]]]])
convT.weight.data = kernel

# Step 4: Apply the transposed convolution
with torch.no_grad():
    y = convT(x)

# Step 5: Print result
print("Transposed convolution output:")
print(y[0, 0])  # Remove batch and channel dimensions