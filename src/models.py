"""
Models module for DustSCAN.
Defines the UNet-based architectures and custom loss functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DustSCANUNet(nn.Module):
    """
    UNet architecture for DustSCAN using an EfficientNet-B4 encoder.
    """
    def __init__(self):
        """
        Initialize the DustSCANUNet model.
        """
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4", 
            encoder_weights="imagenet",           
            in_channels=5,
            classes=1,                      
            activation=None,
            decoder_attention_type="scse"
        )
        
    def forward(self, x):
        """
        Forward pass for the model, handling padding and cropping.
        """
        x_padded = F.pad(x, (0, 27, 0, 12))
        
        out = self.model(x_padded)
        
        return out[:, :, :148, :357]

def build_advanced_unet_model():
    """
    Factory function to build the advanced UNet model.
    """
    return DustSCANUNet()

class FocalDiceBCELoss(nn.Module):
    """
    Custom compound loss combining Focal, Dice, and BCE.
    """
    def __init__(self, alpha=0.90, gamma=2.0, pos_weight=15.0):
        """
        Initialize the loss function parameters.
        """
        super(FocalDiceBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, y_pred_logits, y_true):
        """
        Compute the compound loss.
        """
        pos_weight_tensor = torch.tensor([self.pos_weight], device=y_pred_logits.device)
        
        bce = F.binary_cross_entropy_with_logits(y_pred_logits, y_true, pos_weight=pos_weight_tensor, reduction='none')
        bce_mean = bce.mean()
        
        y_pred = torch.sigmoid(y_pred_logits)
        
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        focal_loss_mean = focal_loss.mean()
        
        smooth = 1e-6
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice_loss = 1.0 - (2. * intersection + smooth) / (union + smooth)
        dice_loss_mean = dice_loss.mean()
        
        return dice_loss_mean + focal_loss_mean + bce_mean
