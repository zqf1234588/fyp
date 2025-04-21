import segmentation_models_pytorch as smp
from models import DualEncoder
import torch
from Segdataloader import *
import torch
def load_model(backbone,att_method,doubleEncoder,encoder_weight,freeze, in_channels, out_channels):
    print("backbone:" ,backbone)
    if doubleEncoder:
            print("-------------------------------")
            print("----model use double encoder---")
            print("-------------------------------")
            model = DualEncoder.DualEncoderUnetPlusPlus(
                encoder_name=backbone,                  # choose encoder, e.g. mobilenet_v2 or efficientnet-b0
                encoder_weights="imagenet",             # use `imagenet` pretrained weights for encoder initialization
                in_channels=in_channels,                # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=out_channels,                   # model output channels (number of classes in your dataset)
                decoder_attention_type= att_method      # SimAM" "cbam etc."
            )
    else:
        print("-------------------------------")
        print("-----model use one encoder----")
        print("-------------------------------")
        model = smp.UnetPlusPlus(
            encoder_name=backbone,                     # choose encoder, e.g. mobilenet_v2 or efficientnet-b0
            encoder_weights=None,                      # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channels,                   # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channels,                      # model output channels (number of classes in your dataset)
            decoder_attention_type= att_method         #"SimAM" "cbam etc."
        )
    
    if encoder_weight is None:
        print("-------------------------------")
        print("------EncoderWeight Random-----")
        print("-------------------------------")
    elif encoder_weight == "imageNet":
        print("-------------------------------")
        print("-----EncoderWeight ImageNet----")
        print("-------------------------------")
    # If using resnet18
    elif encoder_weight == "R18":
        print("-------------------------------") 
        print("-----EncoderWeight CLeanred----")
        print("-------------------------------")
        # load your pre train weight here for encoder resnet18
        checkpoint = torch.load('', weights_only=True)['state_dict']
        state_dict = checkpoint  
        new_state_dict = {k.replace("backbone.", ""): v 
                          for k, v in state_dict.items() 
                          if not k.startswith("fc.") 
                          and not k.startswith("backbone.fc.") 
                          and not k.startswith("projection_heads")}
        model.encoder.load_state_dict(new_state_dict, strict = False)    
     # If using efficient b4
    elif encoder_weight == "E4":
        print("-------------------------------")
        print("-----EncoderWeight CLeanred----")
        print("-------------------------------")
        # load your pre train weight here for encoder efficientnet_b4
        checkpoint = torch.load('/checkpoint_0102.pth.tar', weights_only=True)['state_dict']
        state_dict = checkpoint  
        new_state_dict = {k.replace("backbone.", ""): v 
                          for k, v in state_dict.items() 
                          if not k.startswith("fc.") 
                          and not k.startswith("backbone.fc.") 
                          and not k.startswith("projection_heads")}
        model.encoder.load_state_dict(new_state_dict, strict = False)
      # load your pre train weight here for encoder efficientnet_b5
    elif encoder_weight == "E5":
        print("-------------------------------")
        print("-----EncoderWeight {encoder_weight}---")
        print("-------------------------------")
        
        checkpoint = torch.load('', weights_only=True)['state_dict']
        state_dict = checkpoint  
        new_state_dict = {k.replace("backbone.", ""): v 
                          for k, v in state_dict.items() 
                          if not k.startswith("fc.") 
                          and not k.startswith("backbone.fc.") 
                          and not k.startswith("projection_heads")}
        model.encoder.load_state_dict(new_state_dict, strict = False)      
        
    else:
        print("-------------------------------")
        print("-----Invalid  EncoderWeight----")
        print("-------------------------------")
        exitt(0)

    model.requires_grad_(True)
    # freeze encoder or not
    if freeze == 0:
        print("-------------------------------")
        print("----Encoder Has NOT freezed----")
        print("-------------------------------")

    else:
        print("-------------------------------")
        print("---Encoder Has been  freezed---")
        print("-------------------------------")
        for param in model.encoder.parameters():
            param.requires_grad = False    #only frezze one encoder weights (the first encoder)

    return model
# get datasets from list of dataset names
def get_datasets(dataset_names, mode, resolution, out_channels):
    datasets = {}
    for name in dataset_names:
        if name == 'refuge2':
            datasets['refuge2'] = REFUGE2Base(
                data_root = "./datasets/refuge2", 
                size = resolution, 
                mode=mode,
                num_classes = out_channels
            )
        elif name == "ORIGA":
            datasets['ORIGA'] = ORIGAbase(
                data_root = "./datasets/ORIGA", 
                size = resolution, 
                mode=mode,
                num_classes = out_channels
            )
        elif name == "G1020":
            datasets['G1020'] = G1020base(
                data_root = "./datasets/G1020", 
                size = resolution, 
                mode=mode,
                num_classes = out_channels
            )
        elif name == "RIMONE":
            datasets['RIMONE'] = RIMONEbase(
                data_root = "./datasets/RIMONE", 
                size = resolution, 
                mode=mode,
                num_classes = out_channels
            )
        else:
            assert("invalid val dataset name")
    return datasets
# get datasets loader from list of dataset structures
def get_dataloaders(datasets,train_batch_size):
    dataloaders = {}
    for name,dataset in datasets.items():
        print(len(dataset))
        dataloaders[name] = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=collate_fn,batch_size=train_batch_size)
    return dataloaders

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    mask = mask.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values, 'mask':mask}


    