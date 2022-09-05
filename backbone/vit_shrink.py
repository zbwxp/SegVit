from mmseg.models.backbones.vit import VisionTransformer, TransformerEncoderLayer
from mmseg.models.builder import BACKBONES
import torch
import math
from mmcv.runner import ModuleList

class ShrinkLayer(TransformerEncoderLayer):
    def forward(self, x, shrink=False):
        if shrink:
            x = self.attn(query=self.norm1(x[0]),
                          key=self.norm1(x[1]),
                          value=self.norm1(x[1]),
                          identity=x[0])
        else:
            x = self.attn(self.norm1(x), identity=x)

        x = self.ffn(self.norm2(x), identity=x)
        return x

@BACKBONES.register_module()
class vit_shrink(VisionTransformer):
    def __init__(self,
                 shrink_idx=8,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(vit_shrink, self).__init__(
            img_size,
            patch_size,
            in_channels,
            embed_dims,
            num_layers,
            num_heads,
            mlp_ratio,
            out_indices,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            with_cls_token,
            output_cls_token,
            norm_cfg,
            act_cfg,
            patch_norm,
            final_norm,
            interpolate_mode,
            num_fcs,
            norm_eval,
            with_cp,
            pretrained,
            init_cfg)
        self.shrink_idx = shrink_idx
        del self.layers
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()

        for i in range(num_layers):
            self.layers.append(
                ShrinkLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True))

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            if i == self.shrink_idx:
                n, hw, c = x.shape
                if hw % 2 != 0:
                    x = x[:, 1:]
                h = w = int(math.sqrt(hw))
                x_ = x.transpose(1, 2).reshape(n, c, h, w)
                down_x = x_[:, :, ::2, ::2]
                down_x = down_x.reshape(n, c, hw // 4).transpose(2, 1)
                qkv = (down_x, x)
                x = layer(qkv, shrink=True)
            else:
                x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                # no need to reshape
                # out = out.reshape(B, hw_shape[0], hw_shape[1],
                #                   C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)