import torch
from torch import nn

class BaseComparator(nn.Module):
    def __init__(self, encoders):
        super().__init__()
        self.encoders = encoders.cuda()
        self.grad_enabled = False

    def compare(self, image1_embed, image2_embed, text_embed):
        raise NotImplementedError   
    
    def forward(self, image1, image2, text):
        image1 = image1.cuda()
        image2 = image2.cuda()  
        image1_features = self.encoders.encode_image(image1).cuda()
        image2_features = self.encoders.encode_image(image2).cuda()
        text = self.encoders.tokenize(text).cuda()
        text_features = self.encoders.encode_text(text).cuda()

        return self.compare(image1_features, image2_features, text_features)   
    
class CosineSimilarityComparator(BaseComparator):

    def __init__(self, encoders, normalize=True):
        super().__init__(encoders)
        self.normalize = normalize
    
    def compare(self, image1_features, image2_features, text_features):
        if self.normalize:
            image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
            image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        sim1 = torch.sum(image1_features * text_features, dim=-1)
        sim2 = torch.sum(image2_features * text_features, dim=-1)
        assert torch.allclose(image1_features[0] @ text_features[0], sim1[0], atol=1e-1), f"Error: {image1_features[0] @ text_features[0]} != {sim1[0]}"
        return sim1 - sim2

class ProjectionComparator(BaseComparator):
    
        def __init__(self, encoders, input_size=512, normalize=True):
            super().__init__(encoders)
            self.normalize = normalize
            self.projection = nn.Linear(3 * input_size, 1).cuda().half()
    
        def compare(self, image1_features, image2_features, text_features):
            x = torch.cat([image1_features, image2_features, text_features], dim=-1)
            x = self.projection(x)
            return x
        
class ProjectionComparator(BaseComparator):
    
        def __init__(self, encoders, input_size=512, normalize=True):
            super().__init__(encoders)
            self.normalize = normalize
        
        def make_projection(self):
            raise NotImplementedError
            # self.projection = nn.Linear(3 * input_size, 1).cuda().half()
            # self.projection = nn.Identity().cuda().half()
    
        def compare(self, image1_features, image2_features, text_features):
            x = torch.cat([image1_features, image2_features, text_features], dim=-1)
            x = self.projection(x)
            image1_features = x[:, :512]
            image2_features = x[:, 512:1024]
            text_features = x[:, 1024:]
            if self.normalize:
                image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
                image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim1 = torch.sum(image1_features * text_features, dim=-1)
            sim2 = torch.sum(image2_features * text_features, dim=-1)
            return sim1 - sim2

class LinearProjectionComparator(ProjectionComparator):
    
    def __init__(self, encoders, input_size=512, normalize=True):
        super().__init__(encoders, input_size, normalize)
        self.make_projection(input_size)

    def make_projection(self, input_size):  
        self.projection = nn.Linear(3 * input_size, 3 * input_size).cuda().half()
    
class NonLinearProjectionComparator(ProjectionComparator):
        
    def __init__(self, encoders, input_size=512, normalize=True):
        super().__init__(encoders, input_size, normalize)
        self.make_projection(input_size)

    def make_projection(self, input_size):
        self.projection = nn.Sequential(
            nn.Linear(3 * input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 3*input_size),
        ).cuda().half()
    

class TrainableMLPComparator(BaseComparator):

    def __init__(self, encoders, input_size=512, hidden_size=512, normalize=True):
        super().__init__(encoders)
        self.encoders = encoders.cuda()
        self.hidden_size = hidden_size
        self.normalize = normalize
        self.mlp = nn.Sequential(
            nn.Linear(3 * input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).cuda().half()

    def compare(self, image1_features, image2_features, text_features):
        if self.normalize:
            image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
            image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        x = torch.cat([image1_features, image2_features, text_features], dim=-1)
        return self.mlp(x)
