from utee import misc
print = misc.logger.info
import torch.nn as nn
import torch.nn.functional as F
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
import torch


class LeNet(nn.Module):
    def __init__(self, args, logger, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            QConv2d(3, 6, kernel_size=5, stride=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv1_'),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            QConv2d(6, 16, kernel_size=5, stride=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv2_'),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            QLinear(16 * 5 * 5, 120, logger=logger, wl_input=args.wl_activate, wl_activate=-1,
                    wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                    ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, name='FC1_'),
            nn.Sigmoid(),
            QLinear(120, 84, logger=logger, wl_input=args.wl_activate, wl_activate=-1,
                    wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                    ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, name='FC2_'),
            nn.Sigmoid(),
            QLinear(84, num_classes, logger=logger, wl_input=args.wl_activate, wl_activate=-1,
                    wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                    ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, name='FC3_')           
            )
        
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)



        return x


class AlexNet(nn.Module):
    def __init__(self, args, logger, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            QConv2d(3, 96, kernel_size=11, stride=4, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv1_'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            QConv2d(96, 256, kernel_size=5, stride=1, padding=2, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv2_'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            QConv2d(256, 384, kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv3_'),
            nn.ReLU(inplace=True),
            QConv2d(384, 384, kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv4_'),
            nn.ReLU(inplace=True),
            QConv2d(384, 256, kernel_size=3, stride=1, padding=1, logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                     onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                     detect=args.detect, target=args.target, name='Conv5_'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            QLinear(256 * 5 * 5, 4096, logger=logger, wl_input=args.wl_activate, wl_activate=-1,
                    wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                    ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, name='FC1_'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            QLinear(4096, 4096, logger=logger, wl_input=args.wl_activate, wl_activate=-1,
                    wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                    ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, name='FC2_'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            QLinear(4096, num_classes, logger=logger, wl_input=args.wl_activate, wl_activate=-1,
                    wl_error=args.wl_error, wl_weight=args.wl_weight, inference=args.inference,
                    onoffratio=args.onoffratio, cellBit=args.cellBit, subArray=args.subArray,
                    ADCprecision=args.ADCprecision, vari=args.vari, t=args.t, v=args.v,
                    detect=args.detect, target=args.target, name='FC3_')
        )

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, args, logger, num_classes=10, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.features = nn.Sequential(
            *self._create_conv_layers(logger, args),
            Inception(192, 64, 96, 128, 16, 32, 32, logger, args),
            Inception(256, 128, 128, 192, 32, 96, 64, logger, args),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Inception(480, 192, 96, 208, 16, 48, 64, logger, args),
            Inception(512, 160, 112, 224, 24, 64, 64, logger, args),
            Inception(512, 128, 128, 256, 24, 64, 64, logger, args),
            Inception(512, 112, 144, 288, 32, 64, 64, logger, args),
            Inception(528, 256, 160, 320, 32, 128, 128, logger, args),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Inception(832, 256, 160, 320, 32, 128, 128, logger, args),
            Inception(832, 384, 192, 384, 48, 128, 128, logger, args),
        )

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            QLinear(1024, num_classes, logger=logger, wl_input=args.wl_activate,
                    wl_activate=-1, wl_error=args.wl_error, wl_weight=args.wl_weight,
                    inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                    subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari,
                    t=args.t, v=args.v, detect=args.detect, target=args.target, name='FC_')
        )

        if init_weights:
            self._initialize_weights()

        print(self.features)
        print(self.classifier)

    def _create_conv_layers(self, logger, args):
        conv_layers = [
            (3, 64, 7, 2, 3, 'Conv1_'),
            (64, 64, 1, 1, 0, 'Conv2_'),
            (64, 192, 3, 1, 1, 'Conv3_')
        ]
        
        layers = []
        for in_channels, out_channels, kernel_size, stride, padding, name in conv_layers:
            layers.append(QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                  padding=padding, logger=logger, wl_input=args.wl_activate,
                                  wl_activate=args.wl_activate, wl_error=args.wl_error, wl_weight=args.wl_weight,
                                  inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                                  subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari,
                                  t=args.t, v=args.v, detect=args.detect, target=args.target, name=name))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        
        return layers

    def forward(self, x):
        x = self.features(x)

        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        if self.training and self.aux_logits:
            return x, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, logger, args):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1, logger=logger, args=args, name='InceptionBranch1_')

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1, logger=logger, args=args, name='InceptionBranch2_1'),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1, logger=logger, args=args, name='InceptionBranch2_2')
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1, logger=logger, args=args, name='InceptionBranch3_1'),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2, logger=logger, args=args, name='InceptionBranch3_2')
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1, logger=logger, args=args, name='InceptionBranch4_')
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, logger, args, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = QConv2d(in_channels, out_channels, logger=logger, wl_input=args.wl_activate,
                            wl_activate=args.wl_activate, wl_error=args.wl_error, wl_weight=args.wl_weight,
                            inference=args.inference, onoffratio=args.onoffratio, cellBit=args.cellBit,
                            subArray=args.subArray, ADCprecision=args.ADCprecision, vari=args.vari,
                            t=args.t, v=args.v, detect=args.detect, target=args.target, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class VGG(nn.Module):
    def __init__(self, args, features, num_classes, logger):
        super(VGG, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        if args.network == "vgg8":
            self.classifier = nn.Sequential(
                QLinear(8192, 1024, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC1_'),
                nn.ReLU(inplace=True),
                QLinear(1024, num_classes, logger=logger,
                        wl_input = args.wl_activate,wl_activate=-1, wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,name='FC2_'))
        elif args.network == "vgg16":
            self.classifier = nn.Sequential(
                QLinear(512 * 7 * 7, 4096, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC1_'),
                nn.ReLU(inplace=True),
                QLinear(4096, 4096, logger=logger,
                        wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, name='FC2_'),
                nn.ReLU(inplace=True),
                QLinear(4096, num_classes, logger=logger,
                        wl_input = args.wl_activate,wl_activate=-1, wl_error=args.wl_error,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,name='FC3_'))

        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, args, logger ):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                             logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                             wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                             subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                             name = 'Conv'+str(i)+'_' )
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
    return nn.Sequential(*layers)


cfg_list = {
    'vgg8': [('C', 128, 3, 'same', 2.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],

    'vgg16': [('C', 64, 3, 'same', 2.0),
                ('C', 64, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 128, 3, 'same', 16.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)]
}


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, logger=None, args=None, block_index=0):
        super(ResNetBlock, self).__init__()
        self.conv1 = QConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight,
                             inference=args.inference, onoffratio=args.onoffratio,
                             cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari,
                             t=args.t, v=args.v, detect=args.detect, target=args.target, 
                             name='Conv1_' + str(block_index) + '_')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(out_channels, out_channels, kernel_size=3, padding=1,
                             logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                             wl_error=args.wl_error, wl_weight=args.wl_weight,
                             inference=args.inference, onoffratio=args.onoffratio,
                             cellBit=args.cellBit, subArray=args.subArray,
                             ADCprecision=args.ADCprecision, vari=args.vari,
                             t=args.t, v=args.v, detect=args.detect, target=args.target, 
                             name='Conv2_' + str(block_index) + '_')
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                QConv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                         logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                         wl_error=args.wl_error, wl_weight=args.wl_weight,
                         inference=args.inference, onoffratio=args.onoffratio,
                         cellBit=args.cellBit, subArray=args.subArray,
                         ADCprecision=args.ADCprecision, vari=args.vari,
                         t=args.t, v=args.v, detect=args.detect, target=args.target, 
                         name='Shortcut_' + str(block_index) + '_'),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, args, logger):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.logger = logger
        self.args = args
        self.features = nn.Sequential(
            QConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                     logger=logger, wl_input=args.wl_activate, wl_activate=args.wl_activate,
                     wl_error=args.wl_error, wl_weight=args.wl_weight,
                     inference=args.inference, onoffratio=args.onoffratio,
                     cellBit=args.cellBit, subArray=args.subArray,
                     ADCprecision=args.ADCprecision, vari=args.vari,
                     t=args.t, v=args.v, detect=args.detect, target=args.target),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, num_blocks[0], block_index=0),
            self._make_layer(block, 128, num_blocks[1], stride=2, block_index=1),
            self._make_layer(block, 256, num_blocks[2], stride=2, block_index=2),
            self._make_layer(block, 512, num_blocks[3], stride=2, block_index=3),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = QLinear(512, num_classes, logger=logger,
                                  wl_input=args.wl_activate, wl_activate=-1, 
                                  wl_error=args.wl_error, wl_weight=args.wl_weight,
                                  inference=args.inference, onoffratio=args.onoffratio,
                                  cellBit=args.cellBit, subArray=args.subArray,
                                  ADCprecision=args.ADCprecision, vari=args.vari,
                                  t=args.t, v=args.v, detect=args.detect, target=args.target,
                                  name='FC1_')
        print(self.features)
        print(self.classifier)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, logger=self.logger, args=self.args, block_index=0))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, logger=self.logger, args=self.args, block_index=i))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#-------------------------------------------------------------------------------#
def lenet(args, logger, pretrained=None):
    model = LeNet(args=args, logger=logger, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def alexnet(args, logger, pretrained=None):
    model = AlexNet(args=args, logger=logger, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def googlenet(args, logger, pretrained=None):
    model = GoogLeNet(args=args, logger=logger, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def vgg8( args, logger, pretrained=None):
    cfg = cfg_list['vgg8']
    layers = make_layers(cfg, args, logger)
    model = VGG(args, layers, num_classes=10, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def vgg16( args, logger, pretrained=None):
    cfg = cfg_list['vgg16']
    layers = make_layers(cfg, args, logger)
    model = VGG(args, layers, num_classes=10, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def resnet18(args, logger, pretrained=None):
    model = ResNet(ResNetBlock, [2, 2, 2, 2], num_classes=10, args=args, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def resnet34(args, logger, pretrained=None):
    model = ResNet(ResNetBlock, [3, 4, 6, 3], num_classes=10, args=args, logger=logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

