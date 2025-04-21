"""
# Training a model

Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
pip install -r requirements.txt
```

## Training

The command to train a  model on a custom dataset (`DATASET_NAME`):

```bash
python train.py --mixed_precision "no" --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" --dataset_name "ISIC" --train_batch_size 4 --gradient_accumulation_steps 4 --gradient_checkpointing --train_data_dir '/root/autodl-tmp/latentSDseg/data' --test_data_dir '/root/autodl-tmp/latentSDseg/data' --in_channels 3 --out_channels 1 --report_to "wandb"

seg_train.py --mixed_precision "no" --train_batch_size 4 --gradient_accumulation_steps 4 --gradient_checkpointing --train_data_dir './datasets/refuge2/train/images' --test_data_dir './datasets/refuge2/test/images' --in_channels 3 --out_channels 3 --report_to "wandb"
```


## Using the 
"""
"""
TODO: fix training mixed precision -- issue with AdamW optimizer
"""

# python seg_train.py --cross_att 0 --doubleEncoder 0 --encoder_weight CL --freeze 0 --resolution 1024 --train_batch_size 4


import argparse
import logging
import math
import os
import shutil
import torch.nn.functional as F
from safetensors.torch import load_file
import random
from pathlib import Path
from diffusers.optimization import get_scheduler
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from Segdataloader import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import scipy.io as sio
# from testdataset import log_validation
from util import *
import torch
from torch import Tensor
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import glob


def iou(outputs:torch.Tensor, labels:torch.Tensor):
    "Calculate Dice coefficient "
    SMOOTH = 1e-6
    intersection = torch.sum(outputs * labels)
    union = torch.sum(outputs) + torch.sum(labels) - intersection
    iou = intersection / union
    return iou.item()



def per_class_dice_loss(input, target, epsilon=1e-5, reduction='mean'):
    """
    Calculate Dice loss for each class separately
    
    Args:
        input (Tensor): Model predictions, expected shape [B, C, H, W]
        target (Tensor): Ground truth, expected shape [B, C, H, W]
        epsilon (float): Small constant to avoid division by zero
        reduction (str): 'none' to return per-class losses, 'mean' to return mean of all classes, 
                         'sum' to return sum of all classes
    
    Returns:
        Tensor: Dice loss (1 - Dice coefficient) for each class or reduced according to reduction parameter
    """
    # Calculate per-class dice coefficients
    dice_coeffs = per_class_dice(input, target, epsilon)
    
    # Convert to loss (1 - dice)
    dice_losses = 1.0 - dice_coeffs
    
    # Apply reduction if specified
    if reduction == 'mean':
        return dice_losses.mean()
    elif reduction == 'sum':
        return dice_losses.sum()
    else:  # 'none'
        return dice_losses

def per_class_dice(input, target, epsilon=1e-5):
    """Calculate Dice coefficient for each class separately"""
    C = input.shape[1]
    dices = []
    for c in range(C):
        pred_c = input[:, c, :, :]
        target_c = target[:, c, :, :]
        inter = 2 * (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dices.append((inter + epsilon) / (union + epsilon))
    return torch.tensor(dices)


@torch.no_grad()
def log_validation(num_classes, test_dataloader, model, accelerator, weight_dtype, name):
    """
    Evaluates the model on a validation dataset and logs IOU and Dice metrics.

    Args:
        num_classes (int): Number of segmentation classes (background, disc, cup).
        test_dataloader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): Trained model to evaluate.
        accelerator (Accelerator or None): Accelerator wrapper used during training, if any.
        weight_dtype (torch.dtype): Data type to cast tensors (e.g., torch.float32 or float16).
        name (str): Name of the validation dataset (for logging).

    Returns:
        Results(dice and iou) and clears CUDA memory.
    """
    logger.info(f"Running validation dataset {name}... ")
    if accelerator!= None:
        model = accelerator.unwrap_model(model)
    # dice_score = 0
    iou_score = 0
    dices = torch.zeros(3)
    for _, sample in enumerate(test_dataloader):
        x = sample["pixel_values"].permute(0,3,1,2).to(weight_dtype)
        mask = sample["mask"].to(weight_dtype)
        reconstructions = model(x)
        mask = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
        reconstructions = F.one_hot(reconstructions.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
        iou_score += iou(reconstructions[:,:-1],mask[:,:-1])
        dices += per_class_dice(reconstructions,mask)
    iou_score /= len(test_dataloader)
    dices /= len(test_dataloader)
    dice_score = dices[:2].mean()
    print(dices)
    del model
    torch.cuda.empty_cache()
    return dice_score, iou_score, dices.tolist()

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a  training script."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        required=False,
        help="model in channels",
    )
    
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        required=False,
        help="model out channels",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    

    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./datasets/refuge2",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # default="./refuge2_weights/GatedAttention_CoordAttention",
        default = "weights/segmentation",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )

    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )    
    
    parser.add_argument(
        "--trainDatasetName",
        type=str,
        default="refuge2",
        help="current dataset for training",
    )
    
    parser.add_argument(
        "--backbone",
        type=str,
        default='timm-efficientnet-b4',
        help=(
            "the encoder backbone"
        ),
    )  
    parser.add_argument(
        "--doubleEncoder",
        type=int,
        default=1,
        help="if using double encoder",
    )
    
    parser.add_argument(
        "--encoder_weight",
        type=str,
        default=None,
        help="load pretrained encoder weights",
    )
    
    parser.add_argument(
        "--freeze",
        type=int,
        default=1,
        help="choose it to freeze one of encoder (the first one)",
    )
    
    parser.add_argument(
        "--fuse_method",
        type=str,
        default = None,
        required=False,
        help="fusing method of features of two encoders"
    )
    
    parser.add_argument(
        "--att_method",
        type=str,
        default = None,
        required=False,
        help="attention method of decoder"
    )

    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    # dataset name lists

    args = parse_args()
    dataset_names = [args.trainDatasetName] # refuge2
    #output dicionary format with encoder setting, encoder weight setting, freeze setting, number of epochs, validation epoch setting, resolution setting
    out_dir = os.path.join(args.output_dir, f"./{args.trainDatasetName}_{args.resolution}/{args.backbone}_{args.encoder_weight}/{args.fuse_method}_{args.att_method}/Encoder_{args.doubleEncoder}_ifCL_{args.encoder_weight}_freeze_{args.freeze}_ep_{args.num_train_epochs}_vep_{args.validation_epochs}_Opt_AdamW_loss_DCE")
    
    logging_dir = os.path.join(out_dir, args.logging_dir)
    try:
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
            print("Deleted weights")
    except Exception as e:
        print(f"Failed to delete directory: {e}")
    else:
        print("first train for these parameters")
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=out_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)

    model = load_model(args.backbone,args.att_method,args.doubleEncoder,args.encoder_weight,args.freeze,args.in_channels,args.out_channels)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    with accelerator.main_process_first():
        # train_dataset = REFUGE2Base(
        #     data_root = args.train_data_dir, 
        #     size = args.resolution, 
        #     mode='train',
        #     num_classes = args.out_channels
        # )
        # print()
        
        
        
        train_datasets = get_datasets(dataset_names, "train", args.resolution, args.out_channels)
        print("Not exit check code!!!")
        val_datasets = get_datasets(dataset_names, "val", args.resolution, args.out_channels)
        
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        mask = torch.stack([example["mask"] for example in examples])
        mask = mask.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, 'mask':mask}

    
    train_dataloaders = get_dataloaders(train_datasets, args.train_batch_size)
    
    val_dataloaders = get_dataloaders(val_datasets, args.train_batch_size)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler
    )


    for key, dataloader in train_dataloaders.items():
        train_dataloaders[key] = accelerator.prepare(dataloader)
    for key, dataloader in val_dataloaders.items():
        val_dataloaders[key] = accelerator.prepare(dataloader)    
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    model.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    data_flag = 0
    current = dataset_names[data_flag]
#     num_update_steps_per_epoch = math.ceil(
#         len(train_dataloader[current]) / args.gradient_accumulation_steps
#     )
#     max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    for name,train_dataset in train_datasets.items():
        logger.info(f"  Num val samples in dataset {name} = {len(train_dataset)}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    for name,val_dataset in val_datasets.items():
        logger.info(f"  Num val samples in dataset {name} = {len(val_dataset)}")

    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )

    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(first_epoch, args.num_train_epochs),
        disable=not accelerator.is_local_main_process,
    )

    progress_bar.set_description("Steps")
    # criterion is dice and CrossEntropy
    criterion = nn.CrossEntropyLoss() if args.out_channels > 1 else nn.BCEWithLogitsLoss()
    best_dice = {}    
    for name in dataset_names:
        best_dice[name] = 0
    early_stop_flag = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        # print current dataset for train
        print("current:", current)
        for step, batch in enumerate(train_dataloaders[current]):
            with accelerator.accumulate(model):
                input_ = batch["pixel_values"].permute(0,3,1,2).to(weight_dtype)
                masks = batch['mask'].to(weight_dtype)
                pred = model(input_)
                loss = criterion(pred, F.one_hot(masks.long(), args.out_channels).permute(0, 3, 1, 2).float())
                loss += per_class_dice_loss(
                    F.softmax(pred, dim=1).float(),
                    F.one_hot(masks.long(), args.out_channels).permute(0, 3, 1, 2).float(),
                )
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # update parameters
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0


            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(**logs)
        progress_bar.update(1)
            
        #run validation
        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                with torch.no_grad():
                    dataloader = val_dataloaders[current]
                    # get validation scores
                    dice, iou, dices = log_validation(args.out_channels, dataloader, model, accelerator, weight_dtype, current)
                    # log resluts in tensorboard
                    accelerator.log({"val dice score": dice}, step=epoch)
                    accelerator.log({"val disc dice score": dices[0]}, step=epoch)
                    accelerator.log({"val cup dice score": dices[1]}, step=epoch)
                    print(f"--current dataset: {current}-")
                    print(f"----- dice score: {dice}-----")
                # if it is best validation then save it
                if dice >= best_dice[current]:
                    copy_best = best_dice[current]
                    best_dice[current] = dice
                    save_path = os.path.join(
                        out_dir, f"checkpoint-{epoch}-{current}-{dice}--{iou}"
                    )
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if dice-copy_best>=0.01:
                        early_stop_flag = 0
                    else:
                        early_stop_flag+=1
                else:
                    early_stop_flag+=1
                if early_stop_flag>=1:
                    data_flag = (data_flag+1)%len(dataset_names)
                    current = dataset_names[data_flag]
    accelerator.end_training()


if __name__ == "__main__":
    main()