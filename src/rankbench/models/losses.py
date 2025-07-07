import torch.nn as nn
import torch.nn.functional as F

TASK_TO_LOSS = {
    'regression': nn.MSELoss(),
    'regression_smoothl1': nn.SmoothL1Loss(),
    'classification': nn.CrossEntropyLoss(),
}

def logistic_pairwise_loss(score_i, score_j, label):
    y = 2 * label.float() - 1
    return F.softplus(-y * (score_i - score_j)).mean()

def pairwise_hinge_loss(score_i, score_j, label, margin=0.01):
    # Convert label to +1 or -1 from 1 and 0
    y = 2 * label.float() - 1  # 1 if i > j, -1 if j > i

    # Compute hinge loss
    loss = F.relu(margin - y * (score_i - score_j))
    return loss.mean()