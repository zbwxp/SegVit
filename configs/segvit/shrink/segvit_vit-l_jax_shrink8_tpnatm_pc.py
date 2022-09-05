_base_ = [
    '../segvit_vit-l_jax_480x480_80k_pc.py'
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
data = dict(samples_per_gpu=4,)