import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from utils.parse_config import parse_model_config
from utils.utils import build_targets


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configurations
    """
    # Get the network hyperparameters from [net] definition
    hyperparams = module_defs.pop(0)
    output_filters = []
    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            in_channels = output_filters[-1] if output_filters else int(
                hyperparams["channels"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            is_pad = int(module_def["pad"])
            padding = (kernel_size - 1) // 2 if is_pad else 0
            batch_normalize = int(module_def.get("batch_normalize", 0))
            bias = not batch_normalize
            activation = module_def["activation"]

            # Add the convolutional layer
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=filters,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             bias=bias)
            modules.add_module("conv_{0}".format(i), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                modules.add_module("batch_norm_{0}".format(i), bn)

            # Add the Activation Layer
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                modules.add_module("leaky_{0}".format(i), activn)

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                # Add the Padding Layer
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)

            # Add the Max Pool Layer
            maxpool = nn.MaxPool2d(
                kernel_size,
                stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            scale_factor = int(module_def["stride"])

            # Add the Upsample Layer
            upsample = nn.Upsample(scale_factor=scale_factor)
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])

            # Add an Empty Layer
            empt = EmptyLayer()
            modules.add_module("route_%d" % i, empt)

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]

            # Add an Empty Layer
            empt = EmptyLayer()
            modules.add_module("shortcut_%d" % i, empt)

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])

            # Add YOLO layer (The Detection Layer)
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

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

        self.mse_loss = nn.MSELoss(
            reduction='elementwise_mean')       # Coordinate loss
        self.bce_loss = nn.BCELoss(
            reduction='elementwise_mean')       # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()    # Class loss

    def forward(self, x, targets=None):
        batch_size = x.size(0)
        num_Grids = x.size(2)
        stride = self.image_dim / num_Grids

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda \
            else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # Output :  Batch_Size *
        #           Num_Anchors *
        #           (4+1+Num_classes) *
        #           Num_Grids *
        #           Num_Grids
        prediction = x.view(batch_size,
                            self.num_anchors,
                            self.bbox_attrs,
                            num_Grids,
                            num_Grids).permute(0, 1, 3, 4, 2).contiguous()

        # Get individual outputs
        pred_x = torch.sigmoid(prediction[..., 0])          # Center x
        pred_y = torch.sigmoid(prediction[..., 1])          # Center y
        pred_w = prediction[..., 2]                         # Width
        pred_h = prediction[..., 3]                         # Height
        pred_conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])       # Cls pred

        # Calculate offsets for each grid
        grid_x = torch.arange(num_Grids).repeat(num_Grids, 1).view(
            [1, 1, num_Grids, num_Grids]).type(FloatTensor)
        grid_y = torch.arange(num_Grids).repeat(num_Grids, 1).t().view(
            [1, 1, num_Grids, num_Grids]).type(FloatTensor)
        scaled_anchors = FloatTensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = grid_x + pred_x.data
        pred_boxes[..., 1] = grid_y + pred_y.data
        pred_boxes[..., 2] = anchor_w * torch.exp(pred_w.data)
        pred_boxes[..., 3] = anchor_h * torch.exp(pred_h.data)

        if targets is not None:
            # Training
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = \
                build_targets(
                    pred_boxes=pred_boxes.cpu().data,
                    pred_conf=pred_conf.cpu().data,
                    pred_cls=pred_cls.cpu().data,
                    target=targets.cpu().data,
                    anchors=scaled_anchors.cpu().data,
                    num_anchors=self.num_anchors,
                    num_classes=self.num_classes,
                    grid_size=num_Grids,
                    ignore_thres=self.ignore_thres,
                    img_dim=self.image_dim,
                )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(pred_x[mask], tx[mask])
            loss_y = self.mse_loss(pred_y[mask], ty[mask])
            loss_w = self.mse_loss(pred_w[mask], tw[mask])
            loss_h = self.mse_loss(pred_h[mask], th[mask])
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_false],
                tconf[conf_mask_false]
            ) + self.bce_loss(
                pred_conf[conf_mask_true],
                tconf[conf_mask_true]
            )
            loss_cls = (1 / batch_size) * self.ce_loss(pred_cls[mask],
                                                       torch.argmax(tcls[mask],
                                                                    1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
        else:
            # Inference
            output = torch.cat(
                (
                    pred_boxes.view(batch_size, -1, 4) * stride,
                    pred_conf.view(batch_size, -1, 1),
                    pred_cls.view(batch_size, 1, self.num_classes)
                ),
                -1
            )

            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h",
                           "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs,
                                                     self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)

                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def save_weights(self, path, cutoff=-1):
        """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff
                          if  cutoff = -1 -> all are saved
        """

        with open(path, "wb") as f:
            self.header_info[3] = self.seen
            self.header_info.tofile(f)

            for i, (module_def, module) in \
                enumerate(zip(self.module_defs[:cutoff],
                              self.module_list[:cutoff])):
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    # If batch norm, load bn first
                    if module_def["batch_normalize"]:
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.data.cpu().numpy().tofile(f)
                        bn_layer.running_var.data.cpu().numpy().tofile(f)
                    # Load conv bias
                    else:
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    # Load conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)

            # Needed to write header when saving weights
            self.header_info = header
            self.seen = header[3]

            # The rest are weights
            weights = np.fromfile(f, dtype=np.float32)

            ptr = 0
            for i, (module_def, module) in enumerate(zip(self.module_defs,
                                                         self.module_list)):
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    if module_def["batch_normalize"]:
                        # Load BN bias, weights, running mean and variance
                        bn_layer = module[1]
                        num_b = bn_layer.bias.numel()  # Number of biases
                        # Bias
                        bn_b = torch.from_numpy(weights[ptr: ptr + num_b]) \
                            .view_as(bn_layer.bias)
                        bn_layer.bias.data.copy_(bn_b)
                        ptr += num_b
                        # Weight
                        bn_w = torch.from_numpy(
                            weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                        bn_layer.weight.data.copy_(bn_w)
                        ptr += num_b
                        # Running Mean
                        bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]) \
                            .view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_rm)
                        ptr += num_b
                        # Running Var
                        bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]) \
                            .view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_rv)
                        ptr += num_b
                    else:
                        # Load conv bias
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr: ptr + num_b]) \
                            .view_as(conv_layer.bias)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b

                    # Load conv weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(
                        weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
