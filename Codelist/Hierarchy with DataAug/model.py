import math
import torch.nn.functional as F
from attention_modules import *

Att_MAP = {'class': 32, 'genus': 16, 'family': 8, 'order': 4}

#################### backbone blocks ####################
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, use_attention=False):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

        if use_attention:
            self.ca = PositionAttention(out_filters)
        self.use_attention = use_attention

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        if self.use_attention:
            x = self.ca(x)
        x += skip

        return x


#################### hierarchy interaction blocks ####################
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class AttentionDohfNeck2(nn.Module):

    def __init__(self, M=32, res_channels=2048, pooling_mode='GAP', add_lambda=0.8):
        super(AttentionDohfNeck2, self).__init__()
        self.M = M
        self.base_channels = res_channels
        self.out_channels = M * res_channels
        self.conv = BasicConv2d(res_channels, self.M, kernel_size=1)

        self.pooling = self.build_pooling(pooling_mode)
        self.EPSILON = 1e-6

        self.add_lambda = add_lambda

    def build_pooling(self, pooling_mode):
        if pooling_mode == "GAP":
            return None
        elif pooling_mode == "GMP":
            return nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("Unknown pooling mode: {}".format(pooling_mode))

    def bilinear_attention_pooling(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pooling is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pooling(features * attentions[:, i:i + 1, ...]).view(B, -1)  # (B, C)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)  # (B, M * C)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        return feature_matrix

    def forward(self, x):
        attention_maps = self.conv(x)
        feature_matrix = self.bilinear_attention_pooling(x, attention_maps)
        return feature_matrix, attention_maps  # (B, M * C), (B, M, AH, AW)

    # CHOF
    def dohf(self, shallow_hiera, deep_hiera):
        """
        from shallow to deep: order, family, genus, class
        shallow_hiera: N, M*C
        deep_hiera: N, M*C
        return
        """
        if shallow_hiera==None:
            return deep_hiera, deep_hiera
        N1, MC1 = shallow_hiera.shape
        M1 = MC1//self.base_channels
        shallow_hiera_mean = shallow_hiera.reshape(N1, M1, self.base_channels)  # N,M1*C -> N,M1,C
        shallow_hiera_mean = shallow_hiera_mean.mean(dim=1)  # N, C

        N2, MC2 = deep_hiera.shape
        M2 = MC2//self.base_channels
        deep_hiera_dohf = deep_hiera.reshape(N2, M2, self.base_channels)  # N,M2*C -> N,M2,C
        deep_hiera_dohf = deep_hiera_dohf.permute(0, 2, 1).contiguous()  # N,M2,C -> N,C,M2

        projection = torch.bmm(shallow_hiera_mean.unsqueeze(1), deep_hiera_dohf)  # N, 1, M2
        projection = torch.bmm(shallow_hiera_mean.unsqueeze(2), projection)  # N, C, M2
        shallow_hiera_norm = torch.norm(shallow_hiera_mean, p=2, dim=1)  # N
        projection = projection / (shallow_hiera_norm * shallow_hiera_norm).view(-1, 1, 1)  # N, C, M2

        orthogonal_comp = deep_hiera_dohf - projection
        deep_hiera_dohf = deep_hiera_dohf + self.add_lambda * orthogonal_comp  # N, C, M2
        deep_hiera_dohf = deep_hiera_dohf.permute(0, 2, 1).contiguous()  # N, C, M2 -> N,M2,C
        deep_hiera_dohf = deep_hiera_dohf.reshape(N2, -1)  # N,M2,C -> N, MC2
        # l2 normalization along dimension M2 and C
        deep_hiera_dohf = F.normalize(deep_hiera_dohf, dim=-1)
        return deep_hiera, deep_hiera_dohf

#################### classify blocks ####################
class ClassifyHead(nn.Sequential):
    def __init__(self, in_channel, out_channel, drop_rate=0, bias=False):
        layers = [nn.Linear(in_channel, out_channel, bias=bias),
                  nn.Softmax(dim=1)]
        if drop_rate > 0:
            layers = [nn.Dropout(p=drop_rate)] + layers
        super().__init__(*layers)


################# hierarchy architecture ####################
class GINN(nn.Module):
    def __init__(self, hierarchy, use_attention=True):   # 128x938x1
        """ Constructor
        Args:
            hierarchy: {'class':100, 'family':47, 'order':18}
            use_attention: if use attentions?
        """
        super(GINN, self).__init__()
        self.hierarchy = hierarchy
        self.hier_names = list(hierarchy.keys())
        self.hierarchical_depth = len(hierarchy)
        self.use_attention = use_attention

        self.feature_embedding = nn.Sequential(
            # Entry flow
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),  # (128+2*0-3)/2(向下取整)+1, 63x468x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),  # 31x233x64
            nn.BatchNorm2d(64),
            Block(64, 128, 2, 2, start_with_relu=False, grow_first=True, use_attention=False),
            Block(128, 256, 2, 2, start_with_relu=True, grow_first=True, use_attention=False),
            Block(256, 728, 2, 2, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False, use_attention=self.use_attention)


            )

        self.hier_branch = {}
        # class, (genera), family, order
        for hier in self.hier_names:
            hier_stage = nn.Sequential(
                SeparableConv2d(1024, 1536, 3, 1, 1),
                nn.BatchNorm2d(1536),
                nn.ReLU(inplace=True),
                SeparableConv2d(1536, 2048, 3, 1, 1),
                nn.BatchNorm2d(2048)
                )
            self.add_module(hier + '_branch', hier_stage)
            self.hier_branch[hier] = hier_stage

        self.hier_neck = {}
        for hier in self.hier_names:
            hier_stage = AttentionDohfNeck2(M=Att_MAP[hier])
            self.add_module(hier + '_neck', hier_stage)
            self.hier_neck[hier] = hier_stage

        self.hier_classifyhead = {}
        for hier in self.hier_names:
            hier_stage = ClassifyHead(in_channel=2048*Att_MAP[hier], out_channel=int(self.hierarchy[hier]), drop_rate=0.5)
            self.add_module(hier + '_classifyHead', hier_stage)
            self.hier_classifyhead[hier] = hier_stage

        # add attention regularization with center loss for each hierarchy
        for hier, category_num in self.hierarchy.items():
            self.register_buffer(
                hier+'_feature_center',
                torch.zeros(category_num, 2048*Att_MAP[hier], requires_grad=False))

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
        # -----------------------------

    def forward(self, x):
        batch_size = x.size(0)
        # trunk
        x = self.feature_embedding(x)  # (B,1024,4,8)
        # branch
        multih_fmap = {}
        for hier in self.hier_names:
            hier_x = x
            hier_x = self.hier_branch[hier](hier_x)
            multih_fmap[hier] = hier_x   # (B,2048,2,8)

        multih_att, multih_atts = {}, {}
        multih_fmatrixs, multih_scores = [], []

        shallow_feature_matrix = None
        for hier in reversed(self.hier_names):
        #for hier in self.hier_names:
            feature_matrix, attention_maps = self.hier_neck[hier](multih_fmap[hier])
            # aggregate single hierarchy feature
            # multih_fmatrixs[hier] = feature_matrix
            shallow_feature_matrix, feature_matrix = self.hier_neck[hier].dohf(shallow_feature_matrix, feature_matrix)
            scores = self.hier_classifyhead[hier](feature_matrix)
            # aggregate dohf hierarchy feature
            multih_fmatrixs.append(feature_matrix)
            multih_scores.append(scores)
            """
            # Generate Attention Map
            if self.training:
                # Randomly choose one of attention maps Ak
                attention_map = []
                for i in range(batch_size):
                    attention_weights = torch.sqrt(abs(attention_maps[i].sum(dim=(1, 2)).detach()) + self.hier_neck[hier].EPSILON)
                    attention_weights = F.normalize(attention_weights, p=1, dim=0)
                    k_index = np.random.choice(self.hier_neck[hier].M, 2, p=attention_weights.cpu().numpy())
                    attention_map.append(attention_maps[i, k_index, ...])
                attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
            else:
                attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

            multih_att[hier] = attention_map
            multih_atts[hier] = attention_maps
            """
        return multih_fmap, multih_fmatrixs[::-1], multih_scores[::-1], multih_att, multih_atts
        #return multih_fmap, multih_fmatrixs, multih_scores, multih_att, multih_atts

################# hierarchy architecture w/o HA ####################
class GINN_woHA(nn.Module):
    def __init__(self, hierarchy, use_attention=True):   # 128x938x1
        """ Constructor
        Args:
            hierarchy: {'class':100, 'family':47, 'order':18}
            use_attention: if use attentions?
        """
        super(GINN_woHA, self).__init__()
        self.hierarchy = hierarchy
        self.hier_names = list(hierarchy.keys())
        self.hierarchical_depth = len(hierarchy)
        self.use_attention = use_attention

        self.feature_embedding = nn.Sequential(
            # Entry flow
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),  # (128+2*0-3)/2(向下取整)+1, 63x468x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),  # 31x233x64
            nn.BatchNorm2d(64),
            Block(64, 128, 2, 2, start_with_relu=False, grow_first=True, use_attention=False),
            Block(128, 256, 2, 2, start_with_relu=True, grow_first=True, use_attention=False),
            Block(256, 728, 2, 2, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True, use_attention=False),
            Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False, use_attention=self.use_attention),
            )

        self.hier_branch = {}
        # class, (genera), family, order
        for hier in self.hier_names:
            hier_stage = nn.Sequential(
                SeparableConv2d(1024, 1536, 3, 1, 1),
                nn.BatchNorm2d(1536),
                nn.ReLU(inplace=True),
                SeparableConv2d(1536, 2048, 3, 1, 1),
                nn.BatchNorm2d(2048)
                )
            self.add_module(hier + '_branch', hier_stage)
            self.hier_branch[hier] = hier_stage

        self.hier_neck = {}
        for hier in self.hier_names:
            hier_stage = AttentionDohfNeck2(M=1)
            self.add_module(hier + '_neck', hier_stage)
            self.hier_neck[hier] = hier_stage

        self.hier_classifyhead = {}
        for hier in self.hier_names:
            hier_stage = ClassifyHead(in_channel=2048, out_channel=int(self.hierarchy[hier]))
            self.add_module(hier + '_classifyHead', hier_stage)
            self.hier_classifyhead[hier] = hier_stage

        # add attention regularization with center loss for each hierarchy
        for hier, category_num in self.hierarchy.items():
            self.register_buffer(
                hier+'_feature_center',
                torch.zeros(category_num, 2048, requires_grad=False))

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
        # -----------------------------

    def forward(self, x):
        # trunk
        x = self.feature_embedding(x)  # (B,1024,4,8)
        # branch
        multih_fmap = {}
        for hier in self.hier_names:
            hier_x = x
            hier_x = self.hier_branch[hier](hier_x)   # (B,2048,2,8)
            hier_x = F.adaptive_avg_pool2d(hier_x, (1, 1))   # (B,2048,1,1)
            hier_x = hier_x.view(hier_x.size(0), -1)   # (B,2048)
            multih_fmap[hier] = hier_x

        multih_att, multih_atts = {}, {}
        multih_fmatrixs, multih_scores = [], []

        shallow_feature_matrix = None
        for hier in reversed(self.hier_names):
            # aggregate single hierarchy feature
            feature_matrix = multih_fmap[hier]
            shallow_feature_matrix, feature_matrix = self.hier_neck[hier].dohf(shallow_feature_matrix, feature_matrix)
            scores = self.hier_classifyhead[hier](feature_matrix)
            # aggregate dohf hierarchy feature
            multih_fmatrixs.append(feature_matrix)
            multih_scores.append(scores)

        return multih_fmap, multih_fmatrixs[::-1], multih_scores[::-1], multih_att, multih_atts


if __name__ == '__main__':
    model = GINN(hierarchy={'class': 150, 'genus': 122, 'family': 42, 'order': 14}, use_attention=True)  # 50.36M
    para = model.parameters()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    input = torch.randn(16, 3, 128, 431)
    out = model(input)
