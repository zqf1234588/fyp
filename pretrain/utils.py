import os
import shutil

import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Saves the model state to a file. If this is the best model so far,
    it also saves a copy under a special filename for best models.

    Args:
        state (dict): The state of the model to save (e.g., weights, epoch, optimizer).
        is_best (bool): Whether the current model is the best so far.
        filename (str): The filename to save the checkpoint as.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    """
    Saves the configuration arguments to a YAML file in the model checkpoint folder.

    Args:
        model_checkpoints_folder (str): Path to the directory where the config file should be saved.
        args (dict): Dictionary of configuration parameters to save.
    """
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy of the model's predictions with respect to the top-k predictions.

    Args:
        output (torch.Tensor): Model outputs (logits or probabilities) of shape [batch_size, num_classes].
        target (torch.Tensor): Ground truth labels of shape [batch_size].
        topk (tuple): Tuple of k values (e.g., (1, 5)) to compute top-k accuracy.

    Returns:
        List[torch.Tensor]: List of accuracy values for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
