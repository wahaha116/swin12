SwinUnet(
  (swin_unet): SwinTransformerSys(
    (patch_embed): PatchEmbed(
      (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): BasicLayer(
        dim=96, input_resolution=(56, 56), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(7, 7), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(7, 7), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.014)
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(56, 56), dim=96
          (reduction): Linear(in_features=384, out_features=192, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): BasicLayer(
        dim=192, input_resolution=(28, 28), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=192, input_resolution=(28, 28), num_heads=6, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(7, 7), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.029)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=192, input_resolution=(28, 28), num_heads=6, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(7, 7), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.043)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(28, 28), dim=192
          (reduction): Linear(in_features=768, out_features=384, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer(
        dim=384, input_resolution=(14, 14), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.057)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.071)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          input_resolution=(14, 14), dim=384
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer(
        dim=768, input_resolution=(7, 7), depth=2
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=768, input_resolution=(7, 7), num_heads=24, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=768, window_size=(7, 7), num_heads=24
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.086)
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=768, input_resolution=(7, 7), num_heads=24, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=768, window_size=(7, 7), num_heads=24
              (qkv): Linear(in_features=768, out_features=2304, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=768, out_features=768, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.100)
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (layers_up): ModuleList(
      (0): PatchExpand(
        (expand): Linear(in_features=768, out_features=1536, bias=False)
        (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
      (1): BasicLayer_up(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.057)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=384, window_size=(7, 7), num_heads=12
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.071)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (upsample): PatchExpand(
          (expand): Linear(in_features=384, out_features=768, bias=False)
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer_up(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=192, input_resolution=(28, 28), num_heads=6, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(7, 7), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.029)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=192, input_resolution=(28, 28), num_heads=6, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=192, window_size=(7, 7), num_heads=6
              (qkv): Linear(in_features=192, out_features=576, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=192, out_features=192, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.043)
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=192, out_features=768, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=768, out_features=192, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (upsample): PatchExpand(
          (expand): Linear(in_features=192, out_features=384, bias=False)
          (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): BasicLayer_up(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=0, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(7, 7), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=3, mlp_ratio=4.0
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              dim=96, window_size=(7, 7), num_heads=3
              (qkv): Linear(in_features=96, out_features=288, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=96, out_features=96, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.014)
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=96, out_features=384, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=384, out_features=96, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (concat_back_dim): ModuleList(
      (0): Identity()
      (1): Linear(in_features=768, out_features=384, bias=True)
      (2): Linear(in_features=384, out_features=192, bias=True)
      (3): Linear(in_features=192, out_features=96, bias=True)
    )
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (norm_up): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    (up): FinalPatchExpand_X4(
      (expand): Linear(in_features=96, out_features=1536, bias=False)
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (output): Conv2d(96, 1000, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)
