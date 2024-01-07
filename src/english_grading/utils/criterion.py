import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSE(nn.Module):
    """
    Calculates the Root Mean Squared Error (RMSE) loss. 
    """

    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()  # ‍♀️ Calling the parent class's init method

        #  Building blocks for RMSE calculation 
        self.mse = nn.MSELoss(reduction='none')  # Mean Squared Error foundation 
        self.reduction = reduction  #  How to reduce the loss across dimensions
        self.eps = eps  #  A tiny value to prevent numerical instability 

    def forward(self, y_pred, y_true):
        """
        Performs the RMSE calculation. 

        Args:
            y_pred (torch.Tensor): Predicted values. 
            y_true (torch.Tensor): True values. 

        Returns:
            torch.Tensor: Calculated RMSE loss. 
        """

        #  Calculating the MSE first 
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)  # 🪶 Adding a small epsilon for stability

        #  Applying reduction if needed 
        if self.reduction == 'none':
            loss = loss  #  No reduction
        elif self.reduction == 'sum':
            loss = loss.sum()  # ➕ Summing across all elements
        elif self.reduction == 'mean':
            loss = loss.mean()  # ➗ Averaging across all elements

        return loss  #  Returning the final RMSE loss 


#  Meet the Multi-Channel RMSE Master! 
class MCRMSE(nn.Module):

    # ✨ Constructor: Let's set the stage for multi-channel evaluation ✨
    def __init__(self, num_scored=6, weights=None):
        super().__init__()  #  Build upon the foundational Module

        #  Grab a trusty RMSE calculator for precise measurements
        self.rmse = RMSE()

        #  Number of channels receiving scores
        self.num_scored = num_scored

        # ⚖️ Weights for balancing scores (default to equal weighting)
        self.weights = [1/num_scored for _ in range(num_scored)] if weights is None else weights

    #  Forward pass: Time to assess performance across channels! 
    def forward(self, yhat, y):

        #  Initialize the score tracker
        score = 0

        #  Iterate through each channel, carefully evaluating its RMSE
        for i, w in enumerate(self.weights):
            channel_rmse = self.rmse(yhat[:, :, i], y[:, :, i])  #  Measure RMSE for this channel
            score += channel_rmse * w  # ⚖️ Weight and accumulate the score

        #  Return the final, weighted multi-channel RMSE
        return score


#  Focal Loss Class: Balancing Class Imbalance with Style 
class Focal(nn.Module):
    """
    Applies Focal Loss, which down-weights well-classified examples 
    to focus training on hard examples. 
    """

    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        """
        Initializes the Focal Loss module. 

        Args:
            weight (torch.Tensor, optional): A weight tensor for classes. ⚖️
            gamma (float, optional): Focusing parameter. Defaults to 2.0. 
            reduction (str, optional): Reduction type. Defaults to "mean". 
        """

        super().__init__()  # Inherit from parent class 
        self.weight = weight  # Store class weights ⚖️
        self.gamma = gamma  # Store focusing parameter 
        self.reduction = reduction  # Store reduction type 

    def forward(self, input_tensor, target_tensor):
        """
        Calculates the Focal Loss. 

        Args:
            input_tensor (torch.Tensor): Input model predictions. 
            target_tensor (torch.Tensor): Target classes. 

        Returns:
            torch.Tensor: The calculated Focal Loss. 
        """

        # 🪄 Softmax it up for probabilities ✨
        log_prob = F.log_softmax(input_tensor, dim=-1)  # Log probs for stability ⚓️
        prob = torch.exp(log_prob)  # Actual probabilities 

        #  Focus on the tricky ones! 
        focal_loss = (1 - prob) ** self.gamma * log_prob  # Down-weight easy examples ⬇️

        #  Target the true classes 
        loss = F.nll_loss(
            focal_loss,
            target_tensor.argmax(dim=1),  # Get true class indices ️
            weight=self.weight,  # Apply class weights if provided ⚖️
            reduction=self.reduction,  # Reduce as specified 
        )

        return loss  # Return the focused loss 


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):
        """
        Calculates the dense cross-entropy loss between predictions and targets. 

        Args:
            x (torch.Tensor): The input tensor containing predictions.  ✨
            target (torch.Tensor): The target tensor containing ground truth labels. 
            weights (torch.Tensor, optional): A tensor of weights to apply to the loss. ⚖️

        Returns:
            torch.Tensor: The mean dense cross-entropy loss. 
        """

        # Cast inputs to float for smooth calculations 
        x = x.float()  
        target = target.float()  

        # Activate the log-softmax function for probability smoothing 
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)  

        # Confront predictions with truth for accountability ⚖️
        loss = -logprobs * target  # ⚔️

        # Aggregate the loss across dimensions for comprehensive assessment 
        loss = loss.sum(-1)  

        # Average the loss for a balanced perspective ☯️
        return loss.mean()  # ⚖️


# ✨ Custom loss function for fine-tuned weighting ✨
class WeightedDenseCrossEntropy(nn.Module):
    def forward(self, x, target, weights=None):

        #  Ensure consistent data types 
        x = x.float()  #  Convert input to float for smooth calculations 
        target = target.float()  #  Align target with float format 

        #  Calculate log probabilities for softmax activation 
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)  #  Apply log_softmax for probability distribution 

        #  Compute the loss with target values 
        loss = -logprobs * target  #  Multiply log probs with target to gauge error 
        loss = loss.sum(-1)  # 🪄 Combine error scores across dimensions 🪄

        # ⚖️ Incorporate weights for tailored adjustments ⚖️
        if weights is not None:
            loss = loss * weights  # ️ Apply weighted emphasis for selective prioritization ️
            loss = loss.sum() / weights.sum()  # ⚖️ Ensure balanced normalization ⚖️
        else:
            loss = loss.mean()  #  Default to averaging for unweighted scenarios 

        return loss  #  Return the calculated loss for model optimization 


def get_criterion(config):
    """
    Fetches the appropriate loss criterion based on the specified configuration,
    ensuring optimal alignment with model training objectives. 

    Args:
        config (dict): A configuration dictionary containing criterion details.

    Returns:
        torch.nn.Module: The requested loss criterion, ready to guide training. ✨
    """

    if config.criterion.criterion_type == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss(
            reduction=config.criterion.smooth_l1_loss.reduction,
            beta=config.criterion.smooth_l1_loss.beta
        )  #  Balances L1 and L2 losses for robust optimization ⚖️

    elif config.criterion.criterion_type == 'RMSE':
        return RMSE(
            eps=config.criterion.rmse_loss.eps,
            reduction=config.criterion.rmse_loss.reduction
        )  #  Measures root mean squared error for regression tasks 

    elif config.criterion.criterion_type == 'MCRMSE':
        return MCRMSE(
            weights=config.criterion.mcrmse_loss.weights,
        )  #  Handles multi-criteria RMSE for diverse objectives 

    # Fallback for unmatched criteria
    return nn.MSELoss()  #  Classic mean squared error for general use cases ▫️
