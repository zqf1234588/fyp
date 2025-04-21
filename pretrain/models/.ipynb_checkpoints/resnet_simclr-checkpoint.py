import torch.nn as nn
import timm
import torch
from segmentation_models_pytorch.encoders import get_encoder
class ResNetSimCLR(nn.Module):
    "model for pre-trained encoder for unet++ segmentation training, contrast learning, encoder is resnet18"
    def __init__(self, base_model, out_dim, num_scales=4):
        super(ResNetSimCLR, self).__init__()
        self.num_scales = num_scales
        self.backbone = self._get_basemodel(base_model, out_dim)
        print(self.backbone)
        # Get the intermediate layers for multi-scale output
        self.intermediate_layers = self._get_intermediate_layers(base_model)
        
        # MLP projection heads for each scale
        self.in_features = [64, 128, 256, self.backbone.fc.in_features]
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_features[i], self.in_features[i]),
                nn.ReLU(),
                nn.Linear(self.in_features[i], out_dim)
            ) for i in range(num_scales)
        ])

    def _get_basemodel(self, model_name, out_dim):
        try:
            model = timm.create_model(
                        model_name=model_name,
                        pretrained=True,
                        num_classes=out_dim
                    )
        except KeyError:
            raise "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
        else:
            return model

    def _get_intermediate_layers(self, model_name):
        # Define which layers to use for multi-scale output
        if "resnet18" in model_name:
            return [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        elif "resnet50" in model_name:
            return [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        else:
            raise "Unsupported model for multi-scale output"

    def forward(self, x):
        # Extract features from intermediate layers
        features = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        for i, layer in enumerate(self.intermediate_layers):
            x = layer(x)
            if i < self.num_scales:
                features.append(x)

        # Apply projection heads to each feature map
        outputs = []
        for i, feature in enumerate(features):
            # import pdb
            # pdb.set_trace()
            # Global average pooling
            pooled_feature = nn.functional.adaptive_avg_pool2d(feature, (1, 1)).view(feature.size(0), -1)
            # Projection head
            output = self.projection_heads[i](pooled_feature)
            outputs.append(output)

        return outputs
    
    
# if __name__=='__main__':
#     model = ResNetSimCLR(base_model="resnet18", out_dim=128, num_scales=4)
#     input_tensor = torch.randn(2, 3, 224, 224)  # Example input
#     outputs = model(input_tensor)
#     for i, output in enumerate(outputs):
#         print(f"Scale {i+1} output shape: {output.shape}")