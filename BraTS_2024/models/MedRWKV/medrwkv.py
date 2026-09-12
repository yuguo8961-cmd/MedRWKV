import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_block import conv_layer, deconv_layer,RWKVBottleneck

class ExplicitBoundaryExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.enhance = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels // 4, 1),
            nn.InstanceNorm3d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        smooth = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)                                 
        high_freq = x - smooth
        edge_map = self.enhance(high_freq)
        return edge_map, high_freq

class BoundaryDrivenCrossDeform(nn.Module):

    def __init__(self, bn_channels, mlpp_channels, dropout_rate=0.2, use_edge_gate=False):
                                                   
        super().__init__()
        self.boundary_extractor = ExplicitBoundaryExtractor(bn_channels)
        self.align = nn.Conv3d(mlpp_channels, bn_channels, 1) if bn_channels != mlpp_channels else nn.Identity()
        self.offset_conv = nn.Sequential(
            nn.Conv3d(bn_channels + 1, bn_channels // 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(bn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(bn_channels // 2, 3, kernel_size=1)                  
        )

                  
        self.deform_scale = nn.Parameter(torch.tensor(0.02))

                                               
        self.fusion = nn.Sequential(
            nn.Conv3d(bn_channels * 2, bn_channels, 1),
            nn.InstanceNorm3d(bn_channels),
            nn.GELU(),
            nn.Dropout3d(p=dropout_rate)
        )

    def _create_grid(self, B, D, H, W, device):
        d = torch.linspace(-1, 1, D, device=device)
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        d, h, w = torch.meshgrid(d, h, w, indexing='ij')
        grid = torch.stack([w, h, d], dim=-1)                
        return grid.unsqueeze(0).expand(B, -1, -1, -1, -1)

    def forward(self, bn_feature, mlpp_feature):
        B, C, D, H, W = bn_feature.shape

                            
        if bn_feature.shape[2:] != mlpp_feature.shape[2:]:
            mlpp_feature = F.interpolate(mlpp_feature, size=(D, H, W), mode='trilinear', align_corners=False)
        mlpp_aligned = self.align(mlpp_feature)

                   
        edge_map, high_freq_feat = self.boundary_extractor(bn_feature)

                         
        offset_input = torch.cat([high_freq_feat, edge_map], dim=1)
        offsets = torch.tanh(self.offset_conv(offset_input)) * self.deform_scale

                     
        grid = self._create_grid(B, D, H, W, bn_feature.device)
        deformed_grid = grid + offsets.permute(0, 2, 3, 4, 1)
        mlpp_deformed = F.grid_sample(
            mlpp_aligned, deformed_grid,
            mode='bilinear', padding_mode='border', align_corners=False
        )
        fused = self.fusion(torch.cat([bn_feature, mlpp_deformed * (1 + edge_map)], dim=1))

                                       
        return bn_feature + fused, edge_map


class mr(nn.Module):
    def __init__(self, res_ratio, in_channels, out_channels,
                 embed_dims=(20, 40, 80, 160, 160), dropout_rate=0.2,
                 deep_supervision=True):
        super().__init__()

                                     
        conv_dim = [in_channels, embed_dims[0], embed_dims[1], embed_dims[2],embed_dims[4]]
                                         

        self.deep_supervision = deep_supervision

        self.conv = conv_layer(
            dim=conv_dim,
            res_ratio=res_ratio,
            dropout_rate=dropout_rate
        )

                                               
        self.rwkv_bottleneck = RWKVBottleneck(
            in_channels=embed_dims[4],       
            hidden_dims=[embed_dims[3], embed_dims[4]],              
                                                 
                                    
            res_ratio=res_ratio,
            dropout_rate=dropout_rate
        )

        self.edge_guided_bottleneck = BoundaryDrivenCrossDeform(
            bn_channels=embed_dims[4],               
            mlpp_channels=embed_dims[4],                       
            dropout_rate=dropout_rate
        )

        self.deconv = deconv_layer(
            embed_dims=(20, 40, 80, 160, 160),                     
            res_ratio=res_ratio,
            dropout_rate=dropout_rate
        )

        self.final_conv = nn.Conv3d(embed_dims[0], out_channels, kernel_size=1)

        if self.deep_supervision:
                          
            self.aux_head0 = nn.Conv3d(embed_dims[4], out_channels, 1)         
            self.aux_head1 = nn.Conv3d(embed_dims[2], out_channels, 1)        
            self.aux_head2 = nn.Conv3d(embed_dims[1], out_channels, 1)        

    def forward(self, x):
        conv_hidden_states = self.conv(x)
        rwkv_outputs = self.rwkv_bottleneck(conv_hidden_states[-1])

        enhanced_bottleneck, edge_map = self.edge_guided_bottleneck(
            bn_feature=conv_hidden_states[-1],         
            mlpp_feature=rwkv_outputs         
        )
        conv_hidden_states = list(conv_hidden_states)
        conv_hidden_states[-1] = enhanced_bottleneck                 

        hidden_states = [rwkv_outputs,conv_hidden_states[3],conv_hidden_states[2],conv_hidden_states[1],conv_hidden_states[0]]
                                                                

        if self.deep_supervision and self.training:
            decoder_outputs = self.deconv(hidden_states, return_intermediate=True)
            logits_main = self.final_conv(decoder_outputs[3])
            logits_aux1 = self.aux_head2(decoder_outputs[2])        
            logits_aux2 = self.aux_head1(decoder_outputs[1])        
            logits_aux3 = self.aux_head0(decoder_outputs[0])
            return logits_main, logits_aux1, logits_aux2, logits_aux3, edge_map

        else:
            u0 = self.deconv(hidden_states, return_intermediate=False)
            logits = self.final_conv(u0)
            return logits



