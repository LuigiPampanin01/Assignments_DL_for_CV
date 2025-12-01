import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the rare class (pothole).
                           If   alpha < 1, it weights the background (class 0) by alpha 
                           and pothole (class 1) by (1 - alpha).
                           Actually, standard usage is often: 
                           alpha for class 1, (1-alpha) for class 0.
            gamma (float): Focusing parameter. Higher gamma (e.g., 2.0 or 4.0) 
                           punishes easy examples more.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] (logits)
        # targets: [B] (class indices)
        
        # 1. Calculate standard Cross Entropy Loss (element-wise)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Get the probabilities of the correct class (p_t)
        pt = torch.exp(-ce_loss)
        
        # 3. Calculate Alpha weighting
        # If target is 1 (pothole), weight = alpha
        # If target is 0 (background), weight = 1 - alpha
        # Note: You can tune alpha. 0.25 is standard for detection background, 
        # but since you want to punish missing potholes, you might try alpha=0.75
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 4. Calculate Focal Loss
        # Formula: -alpha * (1 - pt)^gamma * log(pt)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss