import torch
import torch.nn as nn


# 作用：融合A和x的特征
class nconv(nn.Module):

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, A, x):
        # A是(3N, 3N)  x是(B, C', 3N, 1)  B=C'=64
        x = torch.einsum('vn,bfnt->bfvt', (A, x))  # 后两维相乘，输出(B, C', 3N, 1)
        return x.contiguous()  # 深拷贝后返回


# 作用：将节点特征C'扩大为2C'
class linear(nn.Module):

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # c_in -> c_out 即 64 -> 128
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)  # 即线性变换，x乘一个(128, 64, ?)的矩阵，扩大特征


# 作用：输入输出(3N, B, C')，融合A和x的特征后，提取线性+非线性特征
class gcn_glu(nn.Module):

    def __init__(self, c_in, c_out):
        super(gcn_glu, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, 2 * c_out)  # 输入64 输出128
        self.c_out = c_out  # 64

    def forward(self, x, A):
        # 输入(3N, B, C')
        x = x.unsqueeze(3)  # 新建一个维度(3N, B, C', 1)
        x = x.permute(1, 2, 0, 3)  # 转置成(B, C', 3N, 1)  即(64, 64, 510, 1)
        ax = self.nconv(A, x)  # 传入A和x融合特征，输出不变形的(B, C', 3N, 1)
        axw = self.mlp(ax)  # (B, 2C', 3N, 1)
        axw_1, axw_2 = torch.split(axw, [self.c_out, self.c_out], dim=1)  # 将2C'对半拆开
        axw_new = axw_1 * torch.sigmoid(axw_2)  # (B, C', 3N, 1)  # GLU激活函数，提取线性+非线性特征
        axw_new = axw_new.squeeze(3)  # 消除最后维度(B, C', 3N)
        axw_new = axw_new.permute(2, 0, 1)  # 转置成(3N, B, C')
        return axw_new


# 作用：输入(3N, B, C)，用3个GCN分别融合图结构特征，提取线性+非线性特征后，将3N均分成3份，取最大的1份
# 注意：3N是先按时间分为3块N1，N2，N3，然后每个N里包含170个节点的，不要搞反
class stsgcm(nn.Module):

    def __init__(self, args, num_of_features, output_features_num):
        super(stsgcm, self).__init__()
        c_in = num_of_features  # 64
        c_out = output_features_num  # 64
        num_nodes = args.num_nodes  # 170
        gcn_num = args.gcn_num  # 3
        self.gcn_glu = nn.ModuleList()
        for _ in range(gcn_num):
            self.gcn_glu.append(gcn_glu(c_in, c_out))  # 创建3个带图卷积的GLU激活函数，提取线性/非线性特征，输入输出均为64维
            c_in = c_out  # 第1个输入维可以自定，后续输入必须和上一个输出相等
        self.num_nodes = num_nodes  # 170
        self.gcn_num = gcn_num  # 3

    def forward(self, x, A):
        # x是融合了时空特征的(3N, B, C)，A是乘了mask后的邻接矩阵
        need_concat = []
        for i in range(self.gcn_num):  # 循环3次
            x = self.gcn_glu[i](x, A)  # 输入x,A 输出不变型的(3N, B, C')
            need_concat.append(x)  # 记录输出结果
        need_concat = [i[(self.num_nodes):(2 * self.num_nodes), :, :].unsqueeze(0) for i in  # 3N维取[170,340)，其余维全取
                       need_concat]  # 新增第一个维度，列表中是(1, N, B, C'),(1, N, B, C'),(1, N, B, C')
        outputs = torch.stack(need_concat, dim=0)  # (3, N, B, C')  # 把3N拆成的3部分，堆叠在第一个维度
        outputs = torch.max(outputs, dim=0).values  # (1, N, B, C')  # 只输出3个中总和最大的那个
        return outputs

# class position_embedding(nn.Module):
#     def __init__(self,args):
#         input_length = args.seq_length # T
#         num_of_vertices = args.num_nodes # N
#         embedding_size = args.nhid # C
#         self.temporal_emb = torch.nn.init.xavier_normal_(torch.empty(1, input_length, 1,embedding_size), gain=0.0003).cuda()
#         self.spatial_emb = torch.nn.init.xavier_normal_(torch.empty(1, 1, num_of_vertices, embedding_size), gain=0.0003).cuda()
#         # self.temporal_emb = torch.nn.init.xavier_uniform_(torch.empty(1, input_length, 1, embedding_size), gain=1).cuda()
#         # self.spatial_emb = torch.nn.init.xavier_uniform_(torch.empty(1, 1, num_of_vertices, embedding_size),gain=1).cuda()
#     def forward(self, x):
#         # (B, T, N, C)
#         x = x+self.temporal_emb
#         x = x+self.spatial_emb
#         # (B, T, N, C)
#         return x
