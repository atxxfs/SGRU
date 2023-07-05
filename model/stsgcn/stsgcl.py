import torch
import torch.nn as nn
from model.stsgcn.layers import stsgcm


# 作用：输入(B, T, N, C)，先时间、空间分别嵌入后，每3个时间步采样时间，此时时间维3，空间维N，然后合并时空维度成3N，
#      输入STSGCM中，获取3个时间维中，总和最大的那个。T-2个STSGCM各返回1段，最后拼接回T-2维
#      返回(B, T-2, N, C)
class stsgcl(nn.Module):

    def __init__(self, args, input_length, input_features_num):
        #                         12               64
        super(stsgcl, self).__init__()
        # self.position_embedding = position_embedding(args)
        self.T = input_length  # 12
        self.num_of_vertices = args.num_nodes  # 170
        self.input_features_num = input_features_num  # 64
        output_features_num = args.nhid  # 64
        self.stsgcm = nn.ModuleList()
        for _ in range(self.T - 2):  # 创建10个stsgcm块，输入64维，输出64维
            self.stsgcm.append(stsgcm(args, self.input_features_num, output_features_num))

        # position_embedding
        # Xavier高斯初始化一个(1, 12, 1, 64)矩阵，适应tanh激活函数
        self.temporal_emb = torch.nn.init.xavier_normal_(torch.empty(1, self.T, 1, self.input_features_num),
                                                         gain=0.0003).to(args.device)
        # Xavier高斯初始化一个(1, 1, 170, 64)矩阵，适应tanh激活函数
        self.spatial_emb = torch.nn.init.xavier_normal_(
            torch.empty(1, 1, self.num_of_vertices, self.input_features_num), gain=0.0003).to(args.device)
        # self.temporal_emb = torch.nn.init.xavier_uniform_(torch.empty(1, self.T, 1,self.input_features_num), gain=1).cuda()
        # self.spatial_emb = torch.nn.init.xavier_uniform_(torch.empty(1, 1, self.num_of_vertices, self.input_features_num),gain=1).cuda()

    def forward(self, x, A):
        # x是(B, T, N, C')即(64, 12, 170, 64)
        # A是和mask相乘后的(3N, 3N)即(510, 510)
        x = x + self.temporal_emb  # 时间嵌入，T上不同，N上Broadcast
        x = x + self.spatial_emb  # 空间嵌入，N上不同，T上Broadcast
        data = x
        need_concat = []
        for i in range(self.T - 2):  # 宽度=3的时域特征提取，提取出10个
            # shape is (B, 3, N, C) (64, 3, 170, 64)
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3N, C) (64, 3*170, 64) 时空合并
            t = t.reshape([-1, 3 * self.num_of_vertices, self.input_features_num])
            # shape is (3N, B, C) 转置成(3*170, 64, 64)
            t = t.permute(1, 0, 2)
            # 输入(3N, B, C) 输出(1, N, B, C')
            t = self.stsgcm[i](t, A)
            # 转置成 (B, 1, N, C')，消去1这个维度
            t = t.permute(2, 0, 1, 3).squeeze(1)
            need_concat.append(t)  # 将(B, N, C')，即(64, 170, 64)加入列表，一共循环T-2次（10次）
        outputs = torch.stack(need_concat, dim=1)  # 堆叠(B, T-2, N, C')
        return outputs  # (B, T-2, N, C')
