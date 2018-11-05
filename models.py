# -*- coding: utf-8 -*-
import torch.nn as nn

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configurations
    """
    # Get the network hyperparameters from [net] defination
    hyperparams = module_defs.pop(0)
    input_channels = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    
    for index, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        
        if module_def["type"] == "convolutional":
            in_channels = input_channels[-1]
            out_channels = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            padding = int(module_def["pad"])
            batch_normalize = int(module_def.get("batch_normalize", 0))
            bias = not batch_normalize
            activation = module_def["activation"]
            
            # Add the convolutional layer
            conv = nn.Conv2d(in_channels,
                             out_channels, 
                             kernel_size,
                             stride,
                             padding,
                             bias)
            modules.add_module("conv_{0}".format(index), conv)
            
            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(out_channels)
                modules.add_module("batch_norm_{0}".format(index), bn)
            
            # Add the Activation Layer
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                modules.add_module("leaky_{0}".format(index), activn)
            
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                # Add the Padding Layer
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % index, padding)
            
            # Add the Max Pool Layer
            maxpool = nn.MaxPool2d(
                kernel_size,
                stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % index, maxpool)
            
        elif module_def["type"] == "upsample":
            scale_factor = int(module_def["stride"])
            
            # Add the Upsample Layer
            upsample = nn.Upsample(scale_factor, mode = "bilinear")
            modules.add_module("upsample_%d" % index, upsample)
            
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            in_channels = sum([input_channels[layer_i] for layer_i in layers])
            
            # Add an Empty Layer
            empt = EmptyLayer()
            modules.add_module("route_%d" % index, empt)
            
        elif module_def["type"] == "shortcut":
            in_channels = input_channels[int(module_def["from"])]
            
            # Add an Empty Layer
            empt = EmptyLayer()
            modules.add_module("shortcut_%d" % index, empt)
            
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            
            # Add YOLO layer (The Detection Layer)
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % index, yolo_layer)
        
        module_list.append(modules)
        input_channels.append(in_channels)
        
    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()
        
        
class YOLOLayer(nn.Module):
    """Detection layer"""
    
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

from utils.parse_config import parse_model_config
from pprint import pprint
model_defs = parse_model_config('config/yolov3.cfg')
hyperparams, module_list = create_modules(model_defs)
pprint(hyperparams)
pprint(module_list)
