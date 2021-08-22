import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


__all__ = [
    'VGG', 'vgg11_m4', 'vgg11_m4_bn', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(nn.Module):

    def __init__(self, features: nn.Module, num_classes: 100, init_weights: bool = True) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
}

def _vgg(arch: str, cfg: str, batch_norm: bool, num_classes: int, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes, **kwargs)
    return model


def vgg11_m4(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg11_m4', 'F', False, num_classes=num_classes, **kwargs)

def vgg11_m4_bn(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg11_m4_bn', 'F', True, num_classes=num_classes, **kwargs)

def vgg11(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg11', 'A', False, num_classes=num_classes, **kwargs)

def vgg11_bn(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg11_bn', 'A', True, num_classes=num_classes, **kwargs)

def vgg13(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg13', 'B', False, num_classes=num_classes, **kwargs)

def vgg13_bn(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg13_bn', 'B', True, num_classes=num_classes, **kwargs)

def vgg16(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg16', 'D', False, num_classes=num_classes, **kwargs)

def vgg16_bn(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg16_bn', 'D', True, num_classes=num_classes, **kwargs)

def vgg19(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg19', 'E', False, num_classes=num_classes, **kwargs)

def vgg19_bn(num_classes, **kwargs: Any) -> VGG:
    return _vgg('vgg19_bn', 'E', True, num_classes=num_classes, **kwargs)
