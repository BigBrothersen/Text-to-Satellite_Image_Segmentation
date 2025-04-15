import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(
            self, in_channels=3, num_classes=7, features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.performance = {
            'iou_history': {i: [] for i in range(num_classes)},
            'mIoU_history' : [],
            'epochs' : [],
            'best_IoU' : 0
        }
        # Encoder/Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder/Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def update_performance_history(self, m_iou, c_iou, epochs):
        self.performance['mIoU_history'].append(m_iou)
        self.performance['epochs'].append(epochs)
        if m_iou > self.performance['best_IoU']:
            self.performance['best_IoU'] = m_iou
        for cls, iou in enumerate(c_iou):
            self.performance['iou_history'][cls].append(iou)
    
    def plot_performance(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance['epochs'], self.performance['mIoU_history'], label='mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Model Performance')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_class_IoU(self, class_labels=None):
        plt.figure(figsize=(12, 6))
        if class_labels is None:
            class_labels = {cls: f'Class {cls}' for cls in self.performance['iou_history'].keys()}
        # Plot each class's IoU history
        for cls, iou_history in self.performance['iou_history'].items():
            label = class_labels.get(cls, f'Class {cls}')  # Fallback to numeric if label missing
            plt.plot(self.performance['epochs'], iou_history, label=label)
        
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('Class-wise IoU Over Training')
        plt.grid(True)
        
        # Improved legend placement
        plt.legend(
            bbox_to_anchor=(1.05, 1), 
            loc='upper left', 
            borderaxespad=0.
        )
        
        plt.tight_layout()  # Prevent label clipping
        plt.show()
    
    def forward(self, x):
        skip_connections = []
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse list for upsampling

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling
            skip_connection = skip_connections[idx//2]

            # Handle cases where input dimensions aren't perfectly divisible by 2
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Double convolution

        return self.final_conv(x)