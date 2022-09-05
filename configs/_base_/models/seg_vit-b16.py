checkpoint = './pretrained/vit_base_p16_384_20220308-96dfe169.pth'  # noqa
# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
img_size = 512
in_channels = 768
out_indices = [5, 7, 11]
model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        out_indices=out_indices,
        final_norm=False,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=False,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='ATMHead',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        num_classes=150,
        num_layers=3,
        num_heads=12,
        use_stages=len(out_indices),
        embed_dims=in_channels // 2,
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

find_unused_parameters=True