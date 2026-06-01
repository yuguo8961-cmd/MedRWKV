import torch
import torch.nn as nn
import torch.nn.functional as F
from .RWKV3 import VRWKVLayer

class RWKVBottleneck(nn.Module):

    def __init__(self, in_channels=80, hidden_dims=[80, 160],
                 res_ratio=1.0, dropout_rate=0.2):
        super().__init__()

                      
        is_anisotropic = (res_ratio >= 2 ** 1 + 2 ** 0)                

        if is_anisotropic:
            self.downsample = DownSample(
                in_chns=hidden_dims[1],       
                out_chns=hidden_dims[1],      
                kernels=(1, 2, 2),
                strides=(1, 2, 2)
            )
        else:
            self.downsample = DownSample(
                in_chns=hidden_dims[1],
                out_chns=hidden_dims[1],
                kernels=2,
                strides=2
            )

        self.rwkv_160 = VRWKVLayer(
            n_embd=hidden_dims[1],       
            n_layer=2,               
            layer_id=0,
            channel_gamma=1/6,
            shift_pixel=1,
            drop_path=dropout_rate,
            hidden_rate=4,
            init_mode='fancy',
            init_values=1e-4,
            post_norm=False,
            k_norm=True,
            with_cp=False
        )

    def forward(self, x):
        """"""
        x80 = x
                                     
        x160 = self.rwkv_160(x)
        return x160


class Embed_TriRWKV_Block(nn.Module):

    def __init__(self, dim, dropout_rate=0.2, with_cp=False):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU()
        )

        self.tri_rwkv = VRWKVLayer(
            n_embd=dim,
            n_layer=2,
            layer_id=0,
            channel_gamma=1 / 6,
            shift_pixel=1,
            drop_path=dropout_rate,
            hidden_rate=4,
            init_mode='fancy',
            init_values=1e-4,
            post_norm=False,
            k_norm=True,
            with_cp=with_cp
        )

    def forward(self, x):
        identity = x
        embed_x = self.embedding(x)
        rwkv_out = self.tri_rwkv(embed_x)
        return identity + rwkv_out

class SpatialAwareModulator(nn.Module):

    def __init__(self, src_channels, tgt_channels):
        super().__init__()
              
        self.align_conv = nn.Sequential(
            nn.Conv3d(src_channels, tgt_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(tgt_channels)
        )

        self.attn_net = nn.Sequential(
            nn.Conv3d(tgt_channels * 2, tgt_channels // 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(tgt_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(tgt_channels // 2, tgt_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, src_feat, tgt_feat):
        tgt_shape = tgt_feat.shape[2:]

                                                          
        if src_feat.shape[2:] != tgt_shape:
                                                
            if src_feat.shape[2] < tgt_shape[0] or src_feat.shape[3] < tgt_shape[1]:
                                  
                src_aligned = F.interpolate(src_feat, size=tgt_shape, mode='trilinear', align_corners=False)
            else:
                                                   
                src_aligned = F.adaptive_max_pool3d(src_feat, output_size=tgt_shape)
        else:
            src_aligned = src_feat

                 
        src_aligned = self.align_conv(src_aligned)

                        
        concat_feat = torch.cat([src_aligned, tgt_feat], dim=1)
        spatial_gate = self.attn_net(concat_feat)                              

                        
                                                          
                               
        return tgt_feat * (1 + spatial_gate) + src_aligned


class HierarchicalFeatureInteraction(nn.Module):

    def __init__(self, dim1, dim2, dim3, dim4, reduction=8):
        super().__init__()

                                                  
        self.guide_4to3 = SpatialAwareModulator(dim4, dim3)
        self.guide_3to2 = SpatialAwareModulator(dim3, dim2)
        self.guide_2to1 = SpatialAwareModulator(dim2, dim1)

                                                   
        self.refine_1to2 = SpatialAwareModulator(dim1, dim2)
        self.refine_2to3 = SpatialAwareModulator(dim2, dim3)
        self.refine_3to4 = SpatialAwareModulator(dim3, dim4)

                                      
                                                     
        self.alpha_4to3 = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_3to2 = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_2to1 = nn.Parameter(torch.full((1,), 0.01))

        self.alpha_1to2 = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_2to3 = nn.Parameter(torch.full((1,), 0.01))
        self.alpha_3to4 = nn.Parameter(torch.full((1,), 0.01))

                            
        self.norm1 = nn.Sequential(nn.InstanceNorm3d(dim1), nn.ReLU(inplace=True))
        self.norm2 = nn.Sequential(nn.InstanceNorm3d(dim2), nn.ReLU(inplace=True))
        self.norm3 = nn.Sequential(nn.InstanceNorm3d(dim3), nn.ReLU(inplace=True))
        self.norm4 = nn.Sequential(nn.InstanceNorm3d(dim4), nn.ReLU(inplace=True))

    def forward(self, out1, out2, out3, out4):
                                              
        guide_feat3 = self.guide_4to3(out4, out3)
        out3_guided = self.norm3(out3 + self.alpha_4to3 * guide_feat3)

        guide_feat2 = self.guide_3to2(out3_guided, out2)
        out2_guided = self.norm2(out2 + self.alpha_3to2 * guide_feat2)

        guide_feat1 = self.guide_2to1(out2_guided, out1)
        out1_guided = self.norm1(out1 + self.alpha_2to1 * guide_feat1)

                                              
        refine_feat2 = self.refine_1to2(out1_guided, out2_guided)
        out2_refined = self.norm2(out2_guided + self.alpha_1to2 * refine_feat2)

        refine_feat3 = self.refine_2to3(out2_refined, out3_guided)
        out3_refined = self.norm3(out3_guided + self.alpha_2to3 * refine_feat3)

        refine_feat4 = self.refine_3to4(out3_refined, out4)
        out4_refined = self.norm4(out4 + self.alpha_3to4 * refine_feat4)

        return out1_guided, out2_refined, out3_refined, out4_refined

class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, p=0.2):
                                                  
        super(_ConvINReLU3D, self).__init__()                                 
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),                                          
            nn.Dropout3d(p=p, inplace=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

                                                 
class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class DepthwiseSeparableConv3d(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()

                                
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

                             
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,                         
            bias=False
        )

                                
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSConvINReLU3D(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels,
                                     kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=dropout, inplace=True),             
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DSConvIN3D(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, out_channels,
                                     kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)



class Encoder(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns, out_channels=out_chns, kernel_size=k, padding=p, p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p),
        )
        if in_chns != out_chns:
            self.skip = nn.Conv3d(in_chns, out_chns, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = out + identity
        return self.relu(out)

class Encoder2(nn.Module):

    def __init__(self, in_chns, out_chns, k=3, p=1, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
            DSConvINReLU3D(in_chns, out_chns, k, padding=p, dropout=dropout),
            DSConvIN3D(out_chns, out_chns, k, padding=p),
        )
        self.conv2 = nn.Sequential(
            DSConvINReLU3D(out_chns, out_chns, k, padding=p, dropout=dropout),
            DSConvIN3D(out_chns, out_chns, k, padding=p),
        )

        if in_chns != out_chns:
            self.skip = nn.Conv3d(in_chns, out_chns, 1)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        return self.relu(out + identity)


class DownSample(nn.Module):
    """"""

    def __init__(self, in_chns, out_chns,kernels,strides):
        super(DownSample, self).__init__()


        self.down = nn.Sequential(
                nn.InstanceNorm3d(in_chns),
                nn.Conv3d(in_chns, out_chns, kernel_size=kernels, stride=strides)
            )

    def forward(self, x):
        return self.down(x)


class Decoder(nn.Module):
    def __init__(self, in_chns, out_chns, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            DSConvINReLU3D(in_chns, out_chns, (1, 3, 3), padding=(0, 1, 1), dropout=dropout),
            DSConvIN3D(out_chns, out_chns, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Conv3d(in_chns, out_chns, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x1 + x)

class LightweightUp_sum(nn.Module):

    def __init__(self, in_chns, out_chns, kernel, stride, dropout,
                 attention_block=None, halves=True):
        super().__init__()
        up_chns = in_chns // 2 if halves else in_chns

                   
        self.up = nn.Sequential(
                                   
            nn.ConvTranspose3d(
                in_chns, in_chns,
                kernel_size=kernel,
                stride=stride,
                groups=in_chns,       
                bias=False
            ),
                           
            nn.Conv3d(in_chns, up_chns, 1),
            nn.InstanceNorm3d(up_chns)
        )

        self.attention = attention_block
        self.convs = Decoder(up_chns, out_chns, dropout)

    def forward(self, x1, x2):
        if x2 is not None:
            x_1 = self.up(x1)

                  
            if x_1.shape[2:] != x2.shape[2:]:
                x_1 = F.interpolate(x_1, size=x2.shape[2:],
                                    mode='trilinear', align_corners=False)

            x = x_1 + x2

            if self.attention is not None:
                x, w = self.attention(x)
                x = self.convs(x)
                return x, w
            else:
                x = self.convs(x)
                return x
        else:
            x_1 = self.up(x1)
            x = self.convs(x_1)
            return x


class conv_layer(nn.Module):
    def __init__(self, dim, res_ratio, dropout_rate, input_size=(128, 128, 128)):
        super().__init__()

        self.encoder1 = Encoder(dim[0], dim[1], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)

                                   
                                  
        self.pool1 = DownSample(dim[0], dim[1], kernels=(2, 4, 4), strides=(2, 4, 4))
        self.attn1 = Embed_TriRWKV_Block(
            dim=dim[1],
            dropout_rate=dropout_rate,
            with_cp=False
        )
        self.encoder2 = Encoder(dim[1], dim[2], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)

                                   
                                    
        self.pool2 = DownSample(dim[1], dim[2], kernels=(1, 2, 2), strides=(1, 2, 2))
        self.attn2 = Embed_TriRWKV_Block(
            dim=dim[2],
            dropout_rate=dropout_rate,
            with_cp=False                   
        )
        self.encoder3 = Encoder2(dim[2], dim[3], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)

                                 
                                  
        self.pool3 = DownSample(dim[2], dim[3], kernels=(2, 2, 2), strides=(2, 2, 2))
        self.attn3 = Embed_TriRWKV_Block(
            dim=dim[3],
            dropout_rate=dropout_rate,
            with_cp=False
        )
        self.encoder4 = Encoder2(dim[3], dim[4], k=(1, 3, 3), p=(0, 1, 1), dropout=dropout_rate)

                                      
        self.interaction = HierarchicalFeatureInteraction(
            dim1=dim[1],      
            dim2=dim[2],      
            dim3=dim[3],      
            dim4=dim[4],       
            reduction=8
        )

    def forward(self, x):
        out1 = self.encoder1(x)

                          
        x_branch2 = self.pool1(x)
        x_branch2 = self.attn1(x_branch2)
        out2 = self.encoder2(x_branch2)

                          
        x_branch3 = self.pool2(x_branch2)
        x_branch3 = self.attn2(x_branch3)
        out3 = self.encoder3(x_branch3)

                        
        x_branch4 = self.pool3(x_branch3)
        x_branch4 = self.attn3(x_branch4)
        out4 = self.encoder4(x_branch4)

                                      
        out1_enhanced, out2_enhanced, out3_enhanced, out4_enhanced = self.interaction(out1, out2, out3, out4)

                           
        return [out1_enhanced, out2_enhanced, out3_enhanced, out4_enhanced]


class deconv_layer(nn.Module):
    def __init__(self, embed_dims, res_ratio, dropout_rate):
        super(deconv_layer, self).__init__()
        self.network = nn.ModuleList()

        self.network.append(
            LightweightUp_sum(in_chns=embed_dims[4], out_chns=embed_dims[3],
                              kernel=(1, 1, 1), stride=(1, 1, 1),
                              dropout=dropout_rate, halves=False)
        )

                                                    
                                    
        self.network.append(
            LightweightUp_sum(in_chns=embed_dims[3], out_chns=embed_dims[2],
                              kernel=(2, 2, 2), stride=(2, 2, 2),
                              dropout=dropout_rate, halves=True)
        )

                                                      

        self.network.append(
            LightweightUp_sum(in_chns=embed_dims[2], out_chns=embed_dims[1],
                              kernel=(1, 2, 2), stride=(1, 2, 2),
                              dropout=dropout_rate, halves=True)
        )

                                                         

        self.network.append(
            LightweightUp_sum(in_chns=embed_dims[1], out_chns=embed_dims[0],
                              kernel=(2, 4, 4), stride=(2, 4, 4),
                              dropout=dropout_rate, halves=True)
        )

    def forward(self, hidden_states, return_intermediate=False):
        if return_intermediate:
            outputs = []
            x = self.network[0](hidden_states[0], hidden_states[1])
            outputs.append(x)
            x = self.network[1](x, hidden_states[2])
            outputs.append(x)
            x = self.network[2](x, hidden_states[3])
            outputs.append(x)
            x = self.network[3](x, hidden_states[4])
            outputs.append(x)
            return outputs
        else:
            x = self.network[0](hidden_states[0], hidden_states[1])
            x = self.network[1](x, hidden_states[2])
            x = self.network[2](x, hidden_states[3])
            x = self.network[3](x, hidden_states[4])
            return x



