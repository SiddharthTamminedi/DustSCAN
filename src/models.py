import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def build_advanced_unet_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3", 
        encoder_weights="imagenet",           
        in_channels=5,                  # 3 (dust_rgb) + 1 (sun_zenith) + 1 (pdi)
        classes=1,                      
        activation=None,
        decoder_attention_type="scse"
    )
    return model

class FocalDiceLoss(nn.Module):
    """Focal + Dice loss. plume_id is the curated ground truth -- loss is
    computed over ALL pixels without cloud masking."""
    def __init__(self, alpha=0.85, gamma=2.0, pos_weight=10.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, y_pred_logits, y_true):
        pos_weight_tensor = torch.tensor([self.pos_weight], device=y_pred_logits.device)
        bce = F.binary_cross_entropy_with_logits(y_pred_logits, y_true, pos_weight=pos_weight_tensor, reduction='none')
        y_pred = torch.sigmoid(y_pred_logits)
        
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        focal_loss_mean = focal_loss.mean()
        
        # Dice Loss
        smooth = 1e-6
        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice_loss = 1.0 - (2. * intersection + smooth) / (union + smooth)
        dice_loss_mean = dice_loss.mean()
        
        return focal_loss_mean + dice_loss_mean
