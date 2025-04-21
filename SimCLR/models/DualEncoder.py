import torch
import segmentation_models_pytorch as smp
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlusDecoder


from .GatedF import PixelWiseGatedFusion
from .CrossAttn import WindowCrossAttentionFusion
class DualEncoderUnetPlusPlus(SegmentationModel):
    """Unet++ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Decoder of
    Unet++ is more complex than in usual Unet.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **Unet++**

    Reference:
        https://arxiv.org/abs/1807.10165

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_weights_CL: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        fuse_method: Optional[str] = None,
    ):
        super().__init__()
        self.fuse_method = fuse_method
        if encoder_name.startswith("mit_b"):
            raise ValueError("DUnetPlusPlus is not support encoder_name={}".format(encoder_name))

        self.encoder_1 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights_CL,
        )
        
    
        #The number of encoder output channels
        print(self.fuse_method)
        en_channels = [a for a, b in zip(self.encoder_1.out_channels, self.encoder.out_channels)]
        if self.fuse_method is None:
            en_channels = [c * 2 for c in en_channels]
       
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels = en_channels,
            decoder_channels = decoder_channels,
            n_blocks=encoder_depth,
            # decoder_use_norm=decoder_use_batchnorm,   #####################33
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,           ##################33
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        
        self.fusion_blocks = self.get_fusion_blocks(self.fuse_method)
        
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "DualEconderUnetplusplus-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        if self.fuse_method is None:
            features=[torch.cat((a, b), dim=1) for a, b in zip(self.encoder_1(x), self.encoder(x))]
        else:
            features1 = self.encoder_1(x)
            features2 = self.encoder(x)
            features = [fusion(f1, f2) for f1, f2, fusion in zip(features1, features2, self.fusion_blocks)]
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks
        # if self.classification_head is not None:
        #     labels = self.classification_head(features[-1])

    def get_fusion_blocks(self,method):
        if method == 'gated':
            self.fusion_blocks = nn.ModuleList([
                PixelWiseGatedFusion(c1) for c1, c2 in zip(self.encoder_1.out_channels, self.encoder.out_channels)
            ])               
        elif method == 'a_gated':
            self.fusion_blocks = nn.ModuleList([
                AttentionStyleGatedFusion(c1) for c1, c2 in zip(self.encoder_1.out_channels, self.encoder.out_channels)
            ])
        elif method == 'windowcross':
            self.fusion_blocks = nn.ModuleList([
                CrossAttentionFusion(c1) for c1, c2 in zip(self.encoder_1.out_channels, self.encoder.out_channels)
            ])