import torch
import torch.nn as nn
import timm

class RES50SimCLR(nn.Module):
    def __init__(self, base_model='resnest50d', out_dim=128, num_scales=4):
        "model for pre-trained encoder for unet++ segmentation training, contrast learning, encoder is resnest50d"
        super(RES50SimCLR, self).__init__()
        self.num_scales = num_scales
        
        self.backbone = timm.create_model(base_model, pretrained=False, features_only=True)
        
        # Get the number of feature channels of each layer from backbone
        self.feature_info = self.backbone.feature_info
        self.in_features = [info['num_chs'] for info in self.feature_info][:num_scales]

        # Construct an MLP projection head for each scale
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features[i], self.in_features[i]),
                nn.ReLU(),
                nn.Linear(self.in_features[i], out_dim)
            ) for i in range(num_scales)
        ])

    def forward(self, x):
        # Get the multi-scale feature map output by backbone
        features = self.backbone(x)  # Returns a list containing the feature maps of each scale
        outputs = []
        for i in range(self.num_scales):
            # Adaptively average pool the feature maps of each scale and then send them to the corresponding projection head
            pooled = nn.functional.adaptive_avg_pool2d(features[i], (1, 1)).view(features[i].size(0), -1)
            out = self.projection_heads[i](pooled)
            outputs.append(out)
        return outputs

if __name__ == '__main__':
    model = MiTSimCLR(base_model='mit_b2', out_dim=128, num_scales=4)
    input_tensor = torch.randn(2, 3, 224, 224)
    outputs = model(input_tensor)
    for i, output in enumerate(outputs):
        print(f"Scale {i+1} output shape: {output.shape}")
