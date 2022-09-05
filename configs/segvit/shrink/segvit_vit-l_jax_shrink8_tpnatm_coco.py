_base_ = [
    '../segvit_vit-l_jax_512x512_80k_cocostuff10k.py'
]
out_indices = [7, 23]
model = dict(
    backbone=dict(
        type='vit_shrink',
        shrink_idx=8,
        out_indices=out_indices,
    ),
    decode_head=dict(
        type="TPNATMHead",
        num_layers=3,
        use_stages=len(out_indices),
    )
)
