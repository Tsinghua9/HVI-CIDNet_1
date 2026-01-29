import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.wtconv import hv_fe, i_fe
from net.prior_modules import SoftRegionMask, RegionPooling, RegionPolicyMLP, RegionFiLM, RegionCrossAttention, BoundaryMap, StructureGate
from huggingface_hub import PyTorchModelHubMixin


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],  # 每个阶段的通道数（从浅到深）
                 heads=[1, 2, 4, 8],  # 每个阶段的多头注意力头数
                 norm=False,  # 是否使用 LayerNorm
                 use_wtconv_i=True,
                 use_dwconv_hv=False,
                 fe_type='legacy',
                 lca_type='cab'):
        super(CIDNet, self).__init__()

        # 解包通道数和 head 数量，方便后面使用
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # -----------------------------------------------------------
        #                HV 分支（H 和 V 通道） - Encoder 部分
        # -----------------------------------------------------------
        # 输入是 3 通道的 HVI，其中 HV 分支取前两个通道 (H, V)
        # 初始卷积：3→ch1（比如 3→36）
        # 使用 ReplicationPad2d(1) 是为了保持边缘一致性，卷积后尺寸不变。

        # HV_ways
        # Legacy 3x3 stems (kept for reference):
        # self.HVE_block0 = nn.Sequential(
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        # )
        # self.IE_block0 = nn.Sequential(
        #     nn.ReplicationPad2d(1),
        #     nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        # )

        self.HVE_block0 = hv_fe(ch1, use_dwconv_hv, fe_type=fe_type)
        self.IE_block0 = i_fe(ch1, use_wtconv_i, fe_type=fe_type)
        # 接下来是下采样模块（NormDownsample 定义在 transformer_utils.py）
        # 每经过一个 Downsample，空间尺寸减半、通道数增加
        # eg: (B, 36, 384, 384) → (B, 36, 192, 192)
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # -----------------------------------------------------------
        #                HV 分支（H 和 V 通道） - Decoder 部分
        # -----------------------------------------------------------
        # 上采样模块（NormUpsample 定义在 transformer_utils.py）
        # 每经过一个 Upsample，空间尺寸 ×2，通道数减半
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        # 最后一个输出卷积，把通道数降到 2（恢复 HV 两个通道）
        # 注意：不加激活函数，后续会和 I 分支拼接后再进入 HVI → RGB 逆变换
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        if lca_type == 'cab':
            hv_lca = HV_LCA
            i_lca = I_LCA
        elif lca_type == 'diem':
            hv_lca = DIEMHV_LCA
            i_lca = DIEMI_LCA
        elif lca_type == 'waveformer':
            hv_lca = WaveFormerHV_LCA
            i_lca = WaveFormerI_LCA
        else:
            raise ValueError(f"Unknown lca_type: {lca_type}")

        self.HV_LCA1 = hv_lca(ch2, head2)
        self.HV_LCA2 = hv_lca(ch3, head3)
        self.HV_LCA3 = hv_lca(ch4, head4)
        self.HV_LCA4 = hv_lca(ch4, head4)
        self.HV_LCA5 = hv_lca(ch3, head3)
        self.HV_LCA6 = hv_lca(ch2, head2)

        self.I_LCA1 = i_lca(ch2, head2)
        self.I_LCA2 = i_lca(ch3, head3)
        self.I_LCA3 = i_lca(ch4, head4)
        self.I_LCA4 = i_lca(ch4, head4)
        self.I_LCA5 = i_lca(ch3, head3)
        self.I_LCA6 = i_lca(ch2, head2)

        self.trans = RGB_HVI()

        # Region prior modules (used when index_map is provided)
        self.soft_mask = SoftRegionMask()
        self.region_pool = RegionPooling()
        # stage-1 injection at i_enc2 (channels=ch2)
        self.region_policy = RegionPolicyMLP(ch2)
        self.region_film = RegionFiLM()
        self.region_attn = RegionCrossAttention(ch2, init_alpha=-2.197)
        # Clamp mask routing strength (defaults; can be overridden by train.py CLI flags).
        self.region_attn.mask_bias_scale_max = 1.0
        self.boundary_map = BoundaryMap()
        self.structure_gate = StructureGate(ch2)
        # a=0.10 -> alpha=-2.197
        # a=0.05 -> alpha=-2.944
        # a=0.03 -> alpha=-3.476
        # a=0.02 -> alpha=-3.891
        # a=0.01 -> alpha=-4.595
        # stage-2 injection at i_enc3 (deeper, channels=ch4)
        self.region_policy2 = RegionPolicyMLP(ch4)
        self.region_film2 = RegionFiLM()
        self.region_attn2 = RegionCrossAttention(ch4, init_alpha=-2.197)
        # Deeper stage is more prone to late-training over-injection; keep routing prior milder by default.
        # (Can be overridden by train.py CLI flags.)
        self.region_attn2.mask_bias_scale_max = 0.65
        self.structure_gate2 = StructureGate(ch4)

    def forward(self, x, index_map=None, prior_mode: str = 'gate'):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)  # [B, 3, H, W] -> [H,V,I]
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)  # 抽出 I 通道给 I 分支
        # low
        # Intensity分支
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        # HV分支
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        # 用于 skip connection 的跳连特征
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        # Apply region prior after first cross-branch interaction (safer than pre-LCA)
        if index_map is not None:
            target_hw = i_enc2.shape[-2:]
            if prior_mode == 'film':
                S = self.soft_mask(index_map, target_hw=target_hw)
                V = self.region_pool(i_enc2, S)
                gamma, beta = self.region_policy(V)
                i_enc2 = self.region_film(i_enc2, S, gamma, beta)
            elif prior_mode == 'attn':
                S = self.soft_mask(index_map, target_hw=target_hw)
                V = self.region_pool(i_enc2, S)
                i_enc2 = self.region_attn(i_enc2, S, V)
            else:
                A = self.boundary_map(index_map, target_hw=target_hw)
                i_enc2 = self.structure_gate(i_enc2, A)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        # 按照网络结构图应该是这样：
        i_enc3 = self.IE_block3(i_enc3)   #之前是：i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_3)    #之前是：hv_3 = self.HVE_block3(hv_2)
        # 原来仓库代码：
        # i_enc3 = self.IE_block3(i_enc2)
        # hv_3 = self.HVE_block3(hv_2)

        # Apply a second region prior at a deeper stage (less texture-sensitive, more semantic)
        if index_map is not None:
            target_hw = i_enc3.shape[-2:]
            if prior_mode == 'film':
                S = self.soft_mask(index_map, target_hw=target_hw)
                V = self.region_pool(i_enc3, S)
                gamma, beta = self.region_policy2(V)
                i_enc3 = self.region_film2(i_enc3, S, gamma, beta)
            elif prior_mode == 'attn':
                S = self.soft_mask(index_map, target_hw=target_hw)
                V = self.region_pool(i_enc3, S)
                i_enc3 = self.region_attn2(i_enc3, S, V)
            else:
                A = self.boundary_map(index_map, target_hw=target_hw)
                i_enc3 = self.structure_gate2(i_enc3, A)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        # 按照网络结构图应该是这样：
        i_dec2 = self.ID_block2(i_dec2, v_jump1)#之前是： i_dec2 = self.ID_block2(i_dec3, v_jump1)
        # 原来仓库代码：
        # i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi
