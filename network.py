import torch
import torchvision
import torchvision.models as models

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.shufflenet_v2_x1_0().to(device)
# model = models.resnet101().to(device)
# model = models.mobilenet_v2().to(device)
# summary(model, (3, 256, 256))

m = torchvision.models.resnet50(pretrained=False)
# print([(name, module) for name, module in m.named_children()])
# extract layer1 and layer3, giving as names `feat1` and feat2`
# new_m = torchvision.models._utils.IntermediateLayerGetter(
#     m, {
#         'stage2': 'feat1',
#         'stage3': 'feat2',
#         'stage4': 'feat3',
#         'conv5': 'feat4'
#     })
new_m = torchvision.models._utils.IntermediateLayerGetter(
    m, {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3'
    })
out = new_m(torch.rand(1, 3, 256, 256))
print([(k, v.shape) for k, v in out.items()])
