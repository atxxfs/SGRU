import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x=(B, 32, N, T')  A=(N, N)  -> (B, C, N, T')
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # k=(1,1)二维卷积
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in  # c_in = (2*3+1)*32 =224
        self.mlp = linear(c_in, c_out)  # 224 -> 32
        self.dropout = dropout  # 0.3
        self.order = order  # 2

    def forward(self, x, support):
        # x=(B, 32, N, T')  support=3个(N, N)
        out = [x]
        for a in support:  # 循环3次
            x1 = self.nconv(x, a)  # x与3个A融合邻接特征
            out.append(x1)  # 输出x1=(B, 32, N, T')
            for k in range(2, self.order + 1):  # k=2
                x2 = self.nconv(x1, a)  # x1与A再融合一次特征
                out.append(x2)  # 输出x2
                x1 = x2
        # out总共7个，最开始的x与融合6次后的输出

        h = torch.cat(out, dim=1)  # (B, 224, N, T') 融合后的特征加到特征维上
        h = self.mlp(h)  # (B, 32, N, T')
        h = F.dropout(h, self.dropout, training=self.training)  # 将30%的值丢弃
        return h


class gwnet(nn.Module):

    def __init__(self, args, device=torch.device('cuda:0'), num_nodes=170, dropout=0.3, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):

        super(gwnet, self).__init__()
        self.dropout = dropout  # 0.3
        self.blocks = blocks  # 4
        self.layers = layers  # 2
        self.gcn_bool = gcn_bool  # True
        self.addaptadj = addaptadj  # True

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # 输入2维 输出32维 k=(1, 1)
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # 可学习参数(207, 10)
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                # 可学习参数(10, 207)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        # 循环4次
        for b in range(blocks):
            additional_scope = kernel_size - 1  # 1
            new_dilation = 1  # 1 -> 2 -> 4 -> 8
            # 循环2次
            for i in range(layers):
                # 2维空洞卷积 输入32维 输出32维 k=(1,2) dilation=1代表不膨胀！=2代表间隔为1
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                # 1维空洞门控卷积 输入32维 输出32维 k=(1,2)
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1维残差卷积 输入32维 输出32维
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1维跳跃卷积 输入32维 输出256维
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                # feature=32的批标准化
                self.bn.append(nn.BatchNorm2d(residual_channels))
                # 膨胀卷积，扩大感受野
                new_dilation *= 2
                # 2 -> 4 -> 5 -> 7 -> 8 -> 10 -> 11 -> 13
                receptive_field += additional_scope
                additional_scope *= 2  # 2 -> 1 -> 2 -> 1 -> 2 -> 1 -> 2 -> 1
                if self.gcn_bool:
                    #                     32                 32                 0.3                  1
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        # 2维卷积 输入256 输出512
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1))

        # 2维卷积 输入512 输出12（预测时间步）
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1))

        self.receptive_field = receptive_field

    def forward(self, input):
        input = input.permute(0, 3, 2, 1)
        # (B=64, C=2, N=207, T=13)
        in_len = input.size(3)  # T=13
        if in_len < self.receptive_field:  # 13 < 13 不满足
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))  # 补前置零
        else:
            x = input
        x = self.start_conv(x)  # C从2->32  输出(B, 32, N ,T)
        skip = 0

        # 每次迭代，计算自适应邻接矩阵
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # softmax(relu((207, 10) * (10, 207))) 输出adp=(207, 207)
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # list列表里2个(207, 207)，再加一个(207, 207)，就有3个了
            new_supports = self.supports + [adp]

        # 循环8次
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|        gconv   |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1 skip_conv
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # 原始(B, 32, N, T=13)
            residual = x
            # 8个Conv2d空洞卷积，都是32->32，k=(1,2)，奇数时会T-1，偶数时因为有dilation=(2,2)，所以T-2。所以输出(B, 32, N, T')，T'=12,10,9,7,6,4,3,1
            filter = self.filter_convs[i](residual)
            # tanh激活
            filter = torch.tanh(filter)
            # 8个Conv1d门控卷积，都是32->32，k=(1,)，偶数时有dilation=(2,2)，但因为卷积核为1，所以无效。输出(B, 32, N, T')，T'同上
            gate = self.gate_convs[i](residual)
            # sigmoid激活
            gate = torch.sigmoid(gate)
            # 空洞卷积结果，和门控卷积结果，算Hadamard积
            x = filter * gate
            # 将Hadamard积输出结果，再通过skip卷积
            s = x
            # 8个Conv1d跳跃卷积，都是32->256，k=(1,)  # s=(B, 256, N, T')
            s = self.skip_convs[i](s)
            try:  # 第一次skip空指针报错
                skip = skip[:, :, :,  -s.size(3):]  # skip的T'缩小到和s一致
            except:
                skip = 0
            # 新的skip与s连接
            skip = s + skip
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # x=(B, 32, N, T')  new_supports = 3个(N, N)  输出形状不变
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            # 残差连接，residual的T'下降到GCN输出的T'
            x = x + residual[:, :, :, -x.size(3):]
            # 8个维度=32的BatchNorm2d，让64个Batch做标准化
            x = self.bn[i](x)
        # skip=(B, 256, N, 1)，只输出skip的ReLU
        x = F.relu(skip)
        # 256->512后ReLU
        x = F.relu(self.end_conv_1(x))
        # 512->12
        x = self.end_conv_2(x)
        # 输出(B, 12, N, 1)  # 通过线性变换将对输出的12步预测转移到C维上
        return x
