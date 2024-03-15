import torch
import segmentation_models_pytorch as smp
import os.path as p

from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, vgg16_bn, VGG16_BN_Weights, vgg16, VGG16_Weights

def get_checkpoint(log_name, fold=0, data_percent=1., device='cuda'):
  checkpoint = p.join('runs', log_name, f'fold{fold}', f'best_fold={fold}.pth')
  print('Loading checkpoint from:', checkpoint)
  checkpoint = torch.load(checkpoint, map_location=device)
  return checkpoint

def get_segmentation_model(dataset, device, checkpoint=None) -> torch.nn.Module:
    unet = smp.Unet('resnet18', in_channels=3, classes=1, 
                    activation='sigmoid', decoder_use_batchnorm=True)
    unet = unet.to(device)
    if checkpoint is not None:
      saved_unet = torch.load(checkpoint)
      unet.load_state_dict(saved_unet['model'])
    return unet
