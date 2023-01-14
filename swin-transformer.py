""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030
Code/weights from https://github.com/microsoft/Swin-Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
	"""Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
	This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
	the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
	See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
	changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
	'survival rate' as the argument.
	"""
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
	random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
	random_tensor.floor_()  # binarize
	output = x.div(keep_prob) * random_tensor
	return output


class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""

	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
	"""
	将feature map按照window_size划分成一个个没有重叠的window
	Args:
		x: (B, H, W, C)
		window_size (int): window size(M)
	Returns:
		windows: (num_windows*B, window_size, window_size, C)
	"""
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	print("x.shape",x.shape)	#[1, 8, 7, 8, 7, 128] 矩阵数量，8个窗口，7个heads，
	print("x.permute.shape",x.permute(0, 1, 3, 2, 4, 5).shape)	#[1, 8, 8, 7, 7, 128]

	# permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
	# view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]    	M:窗口尺寸
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
	print("windows.shape",windows.shape)	#[64, 7, 7, 128]
	return windows


def window_reverse(windows, window_size: int, H: int, W: int):
	"""
	将一个个window还原成一个feature map
	Args:
		windows: (num_windows*B, window_size, window_size, C)
		window_size (int): Window size(M)
		H (int): Height of image
		W (int): Width of image
	Returns:
		x: (B, H, W, C)
	"""
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	print("B.shape",B)

	# view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	print("x11111.shape",x.shape)	#[4, 8, 8, 7, 7, 128]   56大小：窗口大小是7，8个窗口

	# permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
	# view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	print("x22222.shape",x.shape)	#[4, 56, 56, 128]

	return x


class PatchEmbed(nn.Module):
	"""
	切图重拍
	这个模块是将输入的图片切成一个个的patch，然后再对每个patch中的像素映射为embed dim维。
	2D Image to Patch Embedding
	"""

	def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
		super().__init__()
		patch_size = (patch_size, patch_size)
		self.patch_size = patch_size
		self.in_chans = in_c
		self.embed_dim = embed_dim
		self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

	def forward(self, x):
		_, _, H, W = x.shape

		# padding
		# 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
		pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
		if pad_input:
			# to pad the last 3 dimensions,
			# (W_left, W_right, H_top,H_bottom, C_front, C_back)
			# 在宽度方向的右侧padding，高度方向的底部padding
			x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
						  0, self.patch_size[0] - H % self.patch_size[0],
						  0, 0))

		# 下采样patch_size倍
		x = self.proj(x)
		_, _, H, W = x.shape
		# flatten: [B, C, H, W] -> [B, C, HW]
		# transpose: [B, C, HW] -> [B, HW, C]
		x = x.flatten(2).transpose(1, 2)	#4，3136，128  展平   224/4 = 56*56 = 3136
		x = self.norm(x)
		return x, H, W


class PatchMerging(nn.Module):
	r""" Patch Merging Layer.
	Args:
		dim (int): Number of input channels.
		norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
	"""

	def __init__(self, dim, norm_layer=nn.LayerNorm):
		super().__init__()
		self.dim = dim
		self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)		#4dim ---》 2dim
		self.norm = norm_layer(4 * dim)

	def forward(self, x, H, W):		#x，高，宽
		"""
		x: B, H*W, C		h，w = 56
		"""
		B, L, C = x.shape	#1 3136 128 ， 56*56 =3163
		assert L == H * W, "input feature has wrong size"		#L不等于宽高则报错
		print("B, L, C",B, L, C)		#1 3136 128
		x = x.view(B, H, W, C)
		print("x1212341234",x.shape)		#[1, 56, 56, 128]
		# padding
		# 因为宽和高需要降为原来的一半，所以宽高需要是二的整数倍
		# 如果输入feature map的H，W不是2的整数倍，需要进行padding
		pad_input = (H % 2 == 1) or (W % 2 == 1)
		if pad_input:
			# to pad the last 3 dimensions, starting from the last dimension and moving forward.
			# 只pad最后三个维度H,W,C，并且F.pad函数是倒着来的，头两个参数（0,0）代表Channel维度...
			# (C_front, C_back, W_left, W_right, H_top, H_bottom)
			# 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
			x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
		#间隔选择进行通道拼接
		x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
		x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
		x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
		x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
		print("x0,x1,x2,x3",x0.size(),x1.size(),x2.size(),x3.size())#torch.Size([1, 28, 28, 128]) torch.Size([1, 28, 28, 128]) torch.Size([1, 28, 28, 128]) torch.Size([1, 28, 28, 128])
		#拼接
		x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]	#[1, 28, 28, 512]
		print("x.sahpe",x.size())		# [1, 28, 28, 512]	4*C通道变为原来四倍
		x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]	[1, 784, 512]
		print("x.sahpe",x.size())		# [1, 784, 512]

		x = self.norm(x)
		x = self.reduction(x)  # [B, H/2*W/2, 2*C]
		print("x.sahpe",x.size())    # [1, 784, 256]


		return x


class Mlp(nn.Module):
	""" MLP as used in Vision Transformer, MLP-Mixer and related networks
	"""

	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.drop1 = nn.Dropout(drop)
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop2 = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)	#GELU
		x = self.drop1(x)
		x = self.fc2(x)
		x = self.drop2(x)
		return x


class WindowAttention(nn.Module):
	r""" Window based multi-head self attention (W-MSA) module with relative position bias.
	It supports both of shifted and non-shifted window.
	Args:
		dim (int): Number of input channels.
		window_size (tuple[int]): The height and width of the window.
		num_heads (int): Number of attention heads.
		qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
		attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
		proj_drop (float, optional): Dropout ratio of output. Default: 0.0
	"""

	def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # [Mh, Mw]
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		# define a parameter table of relative position bias
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

		# get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(self.window_size[0])
		coords_w = torch.arange(self.window_size[1])
		coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]		生成网格
		coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
		# [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
		relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += self.window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
		relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
		self.register_buffer("relative_position_index", relative_position_index)

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, mask: Optional[torch.Tensor] = None):
		"""
		Args:
			x: input features with shape of (num_windows*B, Mh*Mw, C)
			mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
		"""
		# [batch_size*num_windows, Mh*Mw, total_embed_dim]
		B_, N, C = x.shape
		# qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
		# reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
		# permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
		qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		# [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
		q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)	#qkv：[3,256,4,49,32]  3个矩阵，通过索引取出。256个窗口分别attention，4头注意力机制，每个窗口7*7=49个元素，
		# qkv：[3,256,4,49,32]  3个矩阵，通过索引取出。256个窗口分别attention，4头注意力机制，每个窗口7*7=49个元素，128/4头=32向量个数
		print("q",q)#[256,4,49,32]
		print("k",k)
		print("v",v)
		# transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
		# @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))
		print("attn",attn.shape)#attention计算后[256，4，49，49]

		# relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
		relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
			self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
		attn = attn + relative_position_bias.unsqueeze(0)		#位置编码+注意力机制
		print('attn',attn.shape)

		if mask is not None:
			# mask: [nW, Mh*Mw, Mh*Mw]
			nW = mask.shape[0]  # num_windows
			# attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
			# mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
			attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, N, N)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)

		attn = self.attn_drop(attn)

		# @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
		# transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
		# reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
		x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
		print("x",x)	#[256,49,128]  256个窗口，每个窗口49个像素，128通道
		x = self.proj(x)	#linear
		x = self.proj_drop(x)
		return x


class SwinTransformerBlock(nn.Module):
	r""" Swin Transformer Block.
	Args:
		dim (int): Number of input channels.
		num_heads (int): Number of attention heads.
		window_size (int): Window size.
		shift_size (int): Shift size for SW-MSA.
		mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
		qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
		drop (float, optional): Dropout rate. Default: 0.0
		attn_drop (float, optional): Attention dropout rate. Default: 0.0
		drop_path (float, optional): Stochastic depth rate. Default: 0.0
		act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
		norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
	"""

	def __init__(self, dim, num_heads, window_size=7, shift_size=0,
				 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
				 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio
		# shift_size=0时是W-MSA
		assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

		self.norm1 = norm_layer(dim)
		# W-MSA或SW-MSA模块
		self.attn = WindowAttention(
			dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
			attn_drop=attn_drop, proj_drop=drop)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

	def forward(self, x, attn_mask):
		H, W = self.H, self.W			#当前输入feature map 宽和高，L = hw，实际的HW不知道，所以现在记录一下
		B, L, C = x.shape
		assert L == H * W, "input feature has wrong size"

		shortcut = x
		x = self.norm1(x)
		x = x.view(B, H, W, C)

		# pad feature maps to multiples of window size
		# 把feature map给pad到window size的整数倍
		pad_l = pad_t = 0
		pad_r = (self.window_size - W % self.window_size) % self.window_size
		pad_b = (self.window_size - H % self.window_size) % self.window_size
		x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
		_, Hp, Wp, _ = x.shape

		# cyclic shift			先做W-MSA后做SW-MSA
		if self.shift_size > 0:		#SW-MSA				#shift_size = 3 窗口选择
			shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))		#偏移
		else:						#W-MSA
			shifted_x = x
			attn_mask = None

		# partition windows
		x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
		x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

		# W-MSA/SW-MSA
		attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

		# merge windows
		attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  #将3136个点，转换为-----》 [nW*B, windows_size:Mh, Mw, C]  4个窗口64*4 = 256个窗口，7，7大小 ，128通道

		# window_reverse		还原attention之前维度，方便下一次attention计算
		shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

		# reverse cyclic shift
		if self.shift_size > 0:				# SW-MSA
			x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		else:								# W-MSA
			x = shifted_x

		if pad_r > 0 or pad_b > 0:
			# 把前面pad的数据移除掉
			x = x[:, :H, :W, :].contiguous()

		x = x.view(B, H * W, C)

		# FFN
		x = shortcut + self.drop_path(x)		#残差
		x = x + self.drop_path(self.mlp(self.norm2(x)))		#残差

		return x


class BasicLayer(nn.Module):
	"""
	A basic Swin Transformer layer for one stage.
	Args:
		dim (int): Number of input channels.
		depth (int): Number of blocks.
		num_heads (int): Number of attention heads.
		window_size (int): Local window size.
		mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
		qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
		drop (float, optional): Dropout rate. Default: 0.0
		attn_drop (float, optional): Attention dropout rate. Default: 0.0
		drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
		norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
		downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
		use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
	"""

	def __init__(self, dim, depth, num_heads, window_size,
				 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
				 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
		super().__init__()
		self.dim = dim
		self.depth = depth
		self.window_size = window_size
		self.use_checkpoint = use_checkpoint
		# 在使用SW-MSA时，窗口需要向右向下平移多少个像素，一般是窗口大小window_size除以2（向下取整）
		self.shift_size = window_size // 2

		# build blocks = Swin Transformer Block成对使用
		self.blocks = nn.ModuleList([
			SwinTransformerBlock(
				dim=dim,
				num_heads=num_heads,
				window_size=window_size,
				# 当shift_size=0时表示当前是W-MSA，shift_size=1时表示当前是SW-MSA
				shift_size=0 if (i % 2 == 0) else self.shift_size,
				mlp_ratio=mlp_ratio,
				qkv_bias=qkv_bias,
				drop=drop,
				attn_drop=attn_drop,
				drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
				norm_layer=norm_layer)
			# 每一个stage中block循环的次数
			for i in range(depth)])

		# patch merging layer
		if downsample is not None:
			self.downsample = downsample(dim=dim, norm_layer=norm_layer)
		else:
			self.downsample = None

	def create_mask(self, x, H, W):
		# calculate attention mask for SW-MSA
		# 保证Hp和Wp是window_size的整数倍
		Hp = int(np.ceil(H / self.window_size)) * self.window_size	#28
		Wp = int(np.ceil(W / self.window_size)) * self.window_size	#28
		# 拥有和feature map一样的通道排列顺序，方便后续window_partition
		# 对于切好的windows进行shift，然后重新按照window size切，单个window若其中全是连续的，则直接计算MSA
		# 若有不连续的，则需要mask了
		img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
		h_slices = (slice(0, -self.window_size),	#
					slice(-self.window_size, -self.shift_size),
					slice(-self.shift_size, None))

		w_slices = (slice(0, -self.window_size),
					slice(-self.window_size, -self.shift_size),
					slice(-self.shift_size, None))
		cnt = 0
		for h in h_slices:
			for w in w_slices:
				img_mask[:, h, w, :] = cnt
				cnt += 1

		mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1] 将feature map按照window_size划分成一个个没有重叠的window
		mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]  每一个windiow拉平
		# 通过相减的广播机制来进行扩充，[nW, 1, Mh*Mw]会将最后一个维度的数据复制Mh*Mw次得到[nW, Mh*Mw, Mh*Mw]
		attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]		广播机制
		print("mask_windows.unsqueeze(1),mask_windows.unsqueeze(2)",mask_windows.unsqueeze(1),mask_windows.unsqueeze(2))
		# [nW, Mh*Mw, Mh*Mw]
		# 相减之后对于不等于0的区域写入-100，等于0的区域就是0
		# 这样就得到了该window进行attention时的mask了
		attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
		return attn_mask

	def forward(self, x, H, W):
		#创建mask--》WS-MSA使用
		attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
		#block
		for blk in self.blocks:	#
			blk.H, blk.W = H, W			#添加宽高属性
			if not torch.jit.is_scripting() and self.use_checkpoint:
				x = checkpoint.checkpoint(blk, x, attn_mask)
			else:
				x = blk(x, attn_mask)
		if self.downsample is not None:
			x = self.downsample(x, H, W)
			# 因为downsample方法是patch merging里实现的，H和W若为奇数会进行padding
			# 所以downsample后的H和W的值是，原H、W为奇数时加一除以二，为偶数时=偶数加一除以二再向下取整=偶数除以二。
			H, W = (H + 1) // 2, (W + 1) // 2   #3/2  = 1.5 = 1
			print(H, W)

		return x, H, W


class SwinTransformer(nn.Module):
	r""" Swin Transformer
		A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
		  https://arxiv.org/pdf/2103.14030
	Args:
		patch_size (int | tuple(int)): Patch size. Default: 4, 输入图片一开始下采样4倍
		in_chans (int): Number of input image channels. Default: 3， 输入图片为RGB三通道
		num_classes (int): Number of classes for classification head. Default: 1000，
		embed_dim (int): Patch embedding dimension. Default: 96， stage1输出的通道数C
		depths (tuple(int)): Depth of each Swin Transformer layer. 每一个stage中Swin Transformer Block的重复次数
		num_heads (tuple(int)): Number of attention heads in different layers. 每一个Swin Transformer Block中自注意力的head数
		window_size (int): Window size. Default: 7，W-MSA和SW-MSA默认采用的窗口大小
		mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4， MLP Block中第一个全连接层的输出翻得倍数
		qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
		drop_rate (float): Dropout rate. Default: 0
		attn_drop_rate (float): Attention dropout rate. Default: 0
		drop_path_rate (float): Stochastic depth rate. Default: 0.1， 是从0递增到0.1
		norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
		patch_norm (bool): If True, add normalization after patch embedding. Default: True
		use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
	"""

	def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
				 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
				 window_size=7, mlp_ratio=4., qkv_bias=True,
				 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
				 norm_layer=nn.LayerNorm, patch_norm=True,
				 use_checkpoint=False, **kwargs):
		super().__init__()

		self.num_classes = num_classes
		self.num_layers = len(depths)
		self.embed_dim = embed_dim
		self.patch_norm = patch_norm
		# stage4输出特征矩阵的channels
		self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
		self.mlp_ratio = mlp_ratio

		# split image into non-overlapping patches
		# 将图片划分为一个个没有重叠的patches   切分重拍
		self.patch_embed = PatchEmbed(
			patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
			norm_layer=norm_layer if self.patch_norm else None)
		self.pos_drop = nn.Dropout(p=drop_rate)

		# stochastic depth
		# 每一个Swin Transformer Block中droppath rate
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 针对每一个block所采用的 droprate

		# build layers		创建stage
		self.layers = nn.ModuleList()
		# 遍历循环建立每个stage，i_layer = [0,1,2,3]
		for i_layer in range(self.num_layers):
			# 注意这里构建的stage和论文图中有些差异
			# 这里的stage不包含该stage的patch_merging层，包含的是下个stage的patch_merging
			layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
								depth=depths[i_layer],
								num_heads=num_heads[i_layer],
								window_size=window_size,
								mlp_ratio=self.mlp_ratio,
								qkv_bias=qkv_bias,
								drop=drop_rate,
								attn_drop=attn_drop_rate,
								drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
								norm_layer=norm_layer,
								downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,		#第四个是没有downsample的
								# 判断该BasicLayer有没有下一个stage的patch_merging
								use_checkpoint=use_checkpoint)
			self.layers.append(layers)

		self.norm = norm_layer(self.num_features)
		self.avgpool = nn.AdaptiveAvgPool1d(1)		#宽高1*1
		self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
		self.apply(self._init_weights)		#权重初始化

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def forward(self, x):
		# x: [B, L, C]	#batch，L = H*W ,channel
		print("x11",x.shape)#torch.Size([4,3,224,224])
		x, H, W = self.patch_embed(x)		#下采样四倍
		print("x1",x.shape)#torch.Size([1, 3136, 128])
		x = self.pos_drop(x)				#随机丢失一部分输入
		print("x2",x.shape)#torch.Size([1, 3136, 128])

		for layer in self.layers:		#构建layers ——》stage1，2，3，4
			x, H, W = layer(x, H, W)

		x = self.norm(x)  # [B, L, C]	#batch，L = H*W ,channel
		print("x3",x.shape)
		x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
		print("x4",x.shape)
		x = torch.flatten(x, 1)
		print("x5",x.shape)
		x = self.head(x)
		return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
	# trained ImageNet-1K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=7,
							embed_dim=96,
							depths=(2, 2, 6, 2),
							num_heads=(3, 6, 12, 24),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
	# trained ImageNet-1K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=7,
							embed_dim=96,
							depths=(2, 2, 18, 2),
							num_heads=(3, 6, 12, 24),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
	# trained ImageNet-1K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=7,
							embed_dim=128,
							depths=(2, 2, 18, 2),
							num_heads=(4, 8, 16, 32),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
	# trained ImageNet-1K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=12,
							embed_dim=128,
							depths=(2, 2, 18, 2),
							num_heads=(4, 8, 16, 32),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
	# trained ImageNet-22K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=7,
							embed_dim=128,
							depths=(2, 2, 18, 2),
							num_heads=(4, 8, 16, 32),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
	# trained ImageNet-22K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=12,
							embed_dim=128,
							depths=(2, 2, 18, 2),
							num_heads=(4, 8, 16, 32),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
	# trained ImageNet-22K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=7,
							embed_dim=192,
							depths=(2, 2, 18, 2),
							num_heads=(6, 12, 24, 48),
							num_classes=num_classes,
							**kwargs)
	return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
	# trained ImageNet-22K
	# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
	model = SwinTransformer(in_chans=3,
							patch_size=4,
							window_size=12,
							embed_dim=192,
							depths=(2, 2, 18, 2),
							num_heads=(6, 12, 24, 48),
							num_classes=num_classes,
							**kwargs)
	return model

if __name__ == "__main__":
	import torch as t
	rgb = t.randn(4,3,224,224)
	model  =swin_base_patch4_window7_224()
	model = model(rgb)
	print(model.shape)

	# from torchstat import stat
	# rgb = t.randn(1,3,512,512)
	# model = swin_base_patch4_window7_224()
	# stat(model, (3, 512, 512))
	# print(stat)

	# #---------------thop-----------------------------thop-------------------
	# from thop import profile
	# model = swin_base_patch4_window7_224()
	# input = t.rand(1,3,512,512)
	# macs,params = profile(model,inputs = (input,))
	# print(macs)
	# print(params)
