import timm
import torch
from torch import nn    

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class TimmModel(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        self.variant = variant
        
        # Print available GPUs for debugging
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            raise RuntimeError("No CUDA devices available")
            
        # Explicitly set parallel processing before model loading
        torch.cuda.set_device(0)  # Set primary GPU
        base_model = timm.create_model(variant, pretrained=pretrained)
        config = resolve_data_config(base_model.default_cfg)
        self.preprocess = create_transform(**config)
        
        # Enable DataParallel only if multiple GPUs are available
        if len(device_ids) > 1:
            self.model = nn.DataParallel(base_model, device_ids=device_ids)
        else:
            self.model = base_model

    def __call__(self):
        return self.model, self.preprocess
    
    def tokenize(self, texts):
        raise NotImplementedError("Tokenization not implemented for TimmModel")

    def postprocess(self, features):
        if self.variant in ["resnet50"]:
            if isinstance(self.model, nn.DataParallel):
                return self.model.module.global_pool(features)
            return self.model.global_pool(features)
        elif self.variant in ["vit_base_patch14_dinov2", "vit_base_patch32_224"]:
            return features[:, 0, :]
        elif self.variant in ["convnextv2_large"]:
            return self.model.head.global_pool(features).squeeze(-1).squeeze(-1)
        else:
            raise ValueError(f"Unsupported variant: {self.variant}")

    def encode_image(self, image):
        features = self.model.forward_features(image)
        features = self.postprocess(features)
        return features
    
    def encode_text(self, text):
        raise NotImplementedError("Text encoding not implemented for TimmModel")

    def unfreeze_image_encoder(self):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.requires_grad_(True)
        else:
            self.model.requires_grad_(True)
        
    def freeze_image_encoder(self):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.requires_grad_(False)
        else:
            self.model.requires_grad_(False)