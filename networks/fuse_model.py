import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch._jit_internal import Optional
from torchvision.models.quantization.utils import _replace_relu, quantize_model
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

#from .networks.deeplab_resnet import Bottleneck, BasicBlock, ResNet, ResNet_locate, ConvertLayer, DeepPoolLayer, ScoreLayer, PoolNet

class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)
        #print(self.downsample) 
        #print('here')
        return out

    def fuse_model(self):

        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)
            



class QuantizableResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs) #
        self.relu0 = nn.ReLU(inplace=False)
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu0(x)
        tmp_x.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x

        #x = self.quant(x)
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        #x = self._forward_impl(x)
        #x = self.dequant(x)
        #return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        #not used/called
        fuse_modules(self, ['conv1', 'bn1', 'relu0'], inplace=True)
        
        for m in self.modules():
            #print(type(m))
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                
                m.fuse_model()
                
        

class QuantizableResNet_locate(ResNet_locate):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet_locate, self).__init__(*args, **kwargs) 

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()



    def forward(self, x):
        
        x = self.quant(x)
        x_size = x.size()[2:]
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        #return xs, infos        
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        #x = self._forward_impl(x)

        x = self.dequant(xs)
        return x, infos

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        # fused this was and model size would be larger. 39.24MB vs 105.98MB from 153.23MB

        if self.ppms:
          for k in range(len(self.ppms)):
              torch.quantization.fuse_modules(self.ppms[k], ['1', '2'], inplace=True)
        if self.ppm_cat:
            torch.quantization.fuse_modules(self.ppm_cat, ['0', '1'], inplace=True)

        if self.infos:
          for k in range(len(self.infos)):
              torch.quantization.fuse_modules(self.infos[k], ['0', '1'], inplace=True)  

        if self.resnet:
              #print(self.resnet)
              torch.quantization.fuse_modules(self.resnet, ['conv1', 'bn1', 'relu'], inplace=True)
              #self.resnet.fuse_model()                            
        for m in self.modules():
            #print(type(m))
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                
                m.fuse_model()
                

def resnet(block=QuantizableBottleneck, layers=[3, 4, 6, 3]):

    model = QuantizableResNet_locate(block, layers)
    model.fuse_model()
    print('---------------------------------------------------------------------------')
    # model = QuantizableResNet(block, layers)
    # model.fuse_model()

    _replace_relu(model)
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    # TODO use pretrained as a string to specify the backend
    backend = 'fbgemm'
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)

    #quantize_model(model, backend)
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)
    return model

model = resnet() #was 39.29mb model

