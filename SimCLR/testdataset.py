import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score

import seaborn as sns
from matplotlib.colors import ListedColormap
from models import DualEncoder
from util import *
from safetensors.torch import load_file
from sklearn.cluster import DBSCAN
"""This file is for testing only"""
class SegmentationTester:
    def __init__(self, model, test_loader, weight_path, num_classes=3, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.class_names = ['disc', 'cup', 'bac']
        self.model.load_state_dict(load_file(weight_path, device=device))
        self.model.eval()
        self.all_images = []
        self.all_preds = []
        self.all_trues = []

    def evaluate(self):
        # Run inference on test set and store predictions, true labels, and images
        self.all_preds = []
        self.all_trues = []
        self.all_images = []
        with torch.no_grad():
            for _, sample in enumerate(self.test_loader):
                images = sample["pixel_values"].permute(0, 3, 1, 2).to(torch.float32).to(self.device)
                masks = sample["mask"].to(torch.float32).to(self.device)
                unique_values, counts = np.unique(masks[0].cpu(), return_counts=True)
                masks = F.one_hot(masks.long(), self.num_classes).permute(0, 3, 1, 2).float()
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)  # [B, H, W]
                true_labels = torch.argmax(masks, dim=1)  # [B, H, W]
                self.all_images.extend(sample["pixel_values"].cpu().numpy())
                self.all_preds.append(preds.cpu().numpy())
                self.all_trues.append(true_labels.cpu().numpy())
        self.all_preds = np.concatenate(self.all_preds, axis=0)
        self.all_trues = np.concatenate(self.all_trues, axis=0)

    def confusion_matrix(self, normalize=True):
        # Compute confusion matrix with optional normalization
        y_true = self.all_trues.flatten()
        y_pred = self.all_preds.flatten()
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        return cm

    def dice_score(self):
        # Calculate Dice score for each class
        dice = []
        for i in range(self.num_classes):
            pred_i = (self.all_preds == i).astype(int)
            true_i = (self.all_trues == i).astype(int)
            intersection = (pred_i * true_i).sum()
            union = pred_i.sum() + true_i.sum()
            dice.append(2 * intersection / (union + 1e-6))
        return dice

    def class_accuracy(self):
        # Compute per-class accuracy from confusion matrix
        cm = self.confusion_matrix()
        acc = []
        for i in range(self.num_classes):
            acc_i = cm[i, i] / (cm[i, :].sum() + 1e-6)
            acc.append(acc_i)
        return acc

    def recall(self):
        # Compute recall for each class
        return recall_score(self.all_trues.flatten(), self.all_preds.flatten(), average=None, labels=list(range(self.num_classes)))

    def precision(self):
        # Compute precision for each class
        return precision_score(self.all_trues.flatten(), self.all_preds.flatten(), average=None, labels=list(range(self.num_classes)))

    def f1(self):
        # Compute F1 score for each class
        return f1_score(self.all_trues.flatten(), self.all_preds.flatten(), average=None, labels=list(range(self.num_classes)))

    def overall_accuracy(self):
        # Compute overall classification accuracy
        return accuracy_score(self.all_trues.flatten(), self.all_preds.flatten())

    def sensitivity(self):
        # Compute sensitivity (recall) from the confusion matrix
        cm = self.confusion_matrix()
        sensitivity = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            sensitivity.append(tp / (tp + fn + 1e-6))
        return sensitivity

    def roc_auc(self):
        # Compute ROC AUC score for each class (one-vs-rest)
        y_true = np.eye(self.num_classes)[self.all_trues.flatten()]
        y_pred = np.eye(self.num_classes)[self.all_preds.flatten()]
        try:
            return roc_auc_score(y_true, y_pred, average=None, multi_class='ovr')
        except:
            return [0.0] * self.num_classes

    def plot_confusion_matrix(self):
        # Use seaborn to draw a confusion matrix in the form of a heat map and save it.
        cm = self.confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".4f", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig('./pic/cm.png')
        # plt.show()
    def visualize_predictions(self, num_samples=5):
        # Sample display and save
        green_red_cmap = ListedColormap(['red', 'green'])
        custom_cmap_pred = ListedColormap(['white', 'black', 'gray'])
        for i in range(min(num_samples, len(self.all_preds))):
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(self.all_images[i], cmap='gray')
            axs[0].set_title('Original Image')
            # Ground Truth
            axs[1].imshow(self.all_trues[i], cmap='gray')
            axs[1].set_title('Ground Truth')
            # Prediction
            axs[2].imshow(self.all_preds[i], cmap='gray')
            axs[2].set_title('Prediction')
            # Correct vs Incorrect
            axs[3].imshow(self.all_trues[i] == self.all_preds[i], cmap=green_red_cmap)
            axs[3].set_title('Correct vs Incorrect')
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'./pic/{i}.png')

    def run_all(self):
        self.evaluate()

        print("Dice Score:")
        for i, v in enumerate(self.dice_score()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nClass Accuracy:")
        for i, v in enumerate(self.class_accuracy()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nPrecision:")
        for i, v in enumerate(self.precision()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nRecall:")
        for i, v in enumerate(self.recall()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nF1 Score:")
        for i, v in enumerate(self.f1()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nSensitivity:")
        for i, v in enumerate(self.sensitivity()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nROC AUC:")
        for i, v in enumerate(self.roc_auc()):
            print(f"  {self.class_names[i]}: {v:.4f}")

        print("\nOverall Accuracy:", f"{self.overall_accuracy():.4f}")

        
        self.visualize_predictions()
        self.plot_confusion_matrix()
if __name__ == '__main__':
    model = DualEncoder.DualEncoderUnetPlusPlus(
        encoder_name='timm-efficientnet-b5',
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        decoder_attention_type= None
    )
    """if using one encoder enable code below"""
#     model = smp.UnetPlusPlus(
#         encoder_name='timm-efficientnet-b5',           # choose encoder, e.g. mobilenet_v2 or efficientnet-b0
#         encoder_weights=None,     # use `imagenet` pretrained weights for encoder initialization
#         in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#         classes=3,                      # model output channels (number of classes in your dataset)
#         # decoder_attention_type= None  #"SimAM"# "cbam"
#     )
        
        
    test_datasets = get_datasets(["refuge2"], "test", 512, 3)
    test_dataloaders = get_dataloaders(test_datasets, 4)
    test_dataloader = test_dataloaders['refuge2']
    weight_path = "your model path"
    test_instance = SegmentationTester(model, test_dataloader, weight_path)
    test_instance.run_all()
