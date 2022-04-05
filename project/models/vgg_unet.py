import torch
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

vgg = torchvision.models.vgg16(pretrained=True)
feature_extractor = create_feature_extractor(vgg, return_nodes={'features.3': 'conv1', 'features.8': 'conv2',
                                                                'features.15': 'conv3', 'features.22': 'conv4'})

out = feature_extractor(torch.rand(1, 3, 256, 256))
print([(k, v.shape) for k, v in out.items()])