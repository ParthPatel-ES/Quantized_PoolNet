import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from torch.nn.quantized import FloatFunctional

#from .deeplab_resnet import resnet50_locate

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []

        self.layerCount = len(list_k[0])

        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

    def fuse_model(self):
        for i in range((self.layerCount)):
            torch.quantization.fuse_modules(self.convert0[i], ['0', '1'], inplace=True)
            #print(self.convert0[i])

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()       
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.q_add00 = FloatFunctional()
        self.q_add01 = FloatFunctional()
        self.q_add02 = FloatFunctional()
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.q_add1 = FloatFunctional()
            self.q_add2 = FloatFunctional() 
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        #for i in range(len(self.pools_sizes)):
           
        y0 = self.convs[0](self.pools[0](x))
        z0 = nn.functional.interpolate(y0, x_size[2:], mode='bilinear', align_corners=True)
        
        y1 = self.convs[1](self.pools[1](x))
        z1 = nn.functional.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True)
        
        y2 = self.convs[2](self.pools[2](x))
        z2 = nn.functional.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True)
        
        resl = self.q_add00.add(resl, z0)
        resl = self.q_add01.add(resl, z1)   
        resl = self.q_add02.add(resl, z2)

        resl = self.relu(resl)

        if self.need_x2:
            resl = nn.functional.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)

        if self.need_fuse:
            resl = self.q_add1.add(resl, x2)
            resl = self.q_add2.add(resl, x3)
            resl = self.conv_sum_c(resl)
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = nn.functional.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


class PoolNet(nn.Module):
    def __init__(self):
        super(PoolNet, self).__init__()
        config = config_resnet
        convert_layers, deep_pool_layers, score_layers = [], [], []
        convert_layers = ConvertLayer(config['convert'])
        test = 15
        self.test = test
        
        for i in range(len(config['deep_pool'][0])):
            deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

        score_layers = ScoreLayer(config['score'])        

        #self.base_model_cfg = 'resnet'
        self.base = ResNet_locate(Bottleneck, [3, 4, 6, 3])

        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        #if self.base_model_cfg == 'resnet':
        self.convert = convert_layers
        # for m in self.modules():
        #   print(type(m))


    # def forward(self, x):
      
    #     x_size = x.size()
    #     conv2merge, infos = self.base(x)
    #     #if self.base_model_cfg == 'resnet':
    #     conv2merge = self.convert(conv2merge)
    #     conv2merge = conv2merge[::-1]

    #     edge_merge = []
    #     merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
    #     for k in range(1, len(conv2merge)-1):
    #         merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])

    #     merge = self.deep_pool[-1](merge)
    #     merge = self.score(merge, x_size)
    #     return merge

    def _forward_impl(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)
        #if self.base_model_cfg == 'resnet':
        conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        edge_merge = []
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])

        merge = self.deep_pool[-1](merge)
        merge = self.score(merge, x_size)
        return merge    

    def forward(self, x):
        return self._forward_impl(x)


                  
class QuantizablePoolNet(PoolNet):
    def __init__(self,*args, **kwargs):
        super(QuantizablePoolNet, self).__init__(*args, **kwargs)

        self.base = QuantizableResNet_locate(QuantizableBottleneck, [3, 4, 6, 3])

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x      

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvertLayer or type(m) == QuantizableResNet_locate:
                  m.fuse_model()        


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

def PNModelfull():
    return PoolNet()

fullPNModel = PNModelfull()
# torch.save(fullPNModel.state_dict(), 'PoolNet1.pth')

model = QuantizablePoolNet()    
model.load_state_dict(torch.load('/content/PoolNet/final.pth', map_location='cpu'))              
quantize_model(model, 'fbgemm')
torch.save(model.state_dict(), 'PoolNet1.pth')
print(model)

def QuantizedPoolNet():
    PNModel = QuantizablePoolNet()    
    PNModel.fuse_model()

    _replace_relu(PNModel)
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    # TODO use pretrained as a string to specify the backend
    backend = 'fbgemm'
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    PNModel.eval()
    # Make sure that weight qconfig matches that of the serialized PNModels
    if backend == 'fbgemm':
        PNModel.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        PNModel.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)

    
    torch.quantization.prepare(PNModel, inplace=True)
    
    #print(type(PNModel))
    PNModel(_dummy_input_data)
    torch.quantization.convert(PNModel, inplace=True)
    #print(PNModel)

    return PNModel

#finalModel = QuantizedPoolNet()    