import torch.nn as nn
import timm
import torch

class EfficientNetSimCLR(nn.Module):
    "model for pre-trained encoder for unet++ segmentation training, contrast learning, encoder is efficient net b4 or b5"
    def __init__(self, base_model='efficientnet-b5', out_dim=128, num_scales=4):
        super(EfficientNetSimCLR, self).__init__()
        self.num_scales = num_scales
        self.backbone = timm.create_model(base_model, pretrained=False, features_only=True)
        
        # Feature info contains channels for each extracted block
        self.feature_info = self.backbone.feature_info
        self.in_features = [info['num_chs'] for info in self.feature_info][:num_scales]

        # MLP projection heads for each scale
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features[i], self.in_features[i]),
                nn.ReLU(),
                nn.Linear(self.in_features[i], out_dim)
            ) for i in range(num_scales)
        ])

    def forward(self, x):
        # Get feature maps from efficientnet blocks
        features = self.backbone(x)  # returns list of feature maps from each stage
        outputs = []
        for i in range(self.num_scales):
            pooled = nn.functional.adaptive_avg_pool2d(features[i], (1, 1)).view(features[i].size(0), -1)
            output = self.projection_heads[i](pooled)
            outputs.append(output)

        return outputs


if __name__ == '__main__':
    # efficient net b4 or b5 change here
    model = EfficientNetSimCLR(base_model='efficientnet_b5', out_dim=128, num_scales=4)
    input_tensor = torch.randn(2, 3, 224, 224)
    outputs = model(input_tensor)
    for i, output in enumerate(outputs):
        print(f"Scale {i+1} output shape: {output.shape}")