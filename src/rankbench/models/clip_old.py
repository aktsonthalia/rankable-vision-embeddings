import clip
import torch
from torch import nn    

class OpenAICLIP(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        
        # Print available GPUs for debugging
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            raise RuntimeError("No CUDA devices available")
            
        # Explicitly set parallel processing before model loading
        torch.cuda.set_device(0)  # Set primary GPU
        base_model, self.preprocess = clip.load(variant, device="cuda", jit=False, pretrained=pretrained)
        # Enable DataParallel only if multiple GPUs are available
        if len(device_ids) > 1:
            self.model = nn.DataParallel(base_model, device_ids=device_ids)
        else:
            self.model = base_model

    def __call__(self):
        return self.model, self.preprocess
    
    def tokenize(self, texts):
        return clip.tokenize(texts).cuda()

    def encode_image(self, image):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.encode_image(image)
        return self.model.encode_image(image)
    
    def encode_text(self, text):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.encode_text(text)
        return self.model.encode_text(text)

    def unfreeze_image_encoder(self):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.visual.requires_grad_(True)
        else:
            self.model.visual.requires_grad_(True)
        
    def freeze_image_encoder(self):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.visual.requires_grad_(False)
        else:
            self.model.visual.requires_grad_(False)