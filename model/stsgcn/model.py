import torch
import torch.nn as nn
from model.stsgcn.stsgcl import stsgcl


# 作用：输入(B, T', N, C)，T'=4，浓缩了T=12的时间特征，然后融合T'与C(64)，成为256维
# 再通过Conv2d降维到128，再通过Conv2d降维到1，输出(B, 1, N)，进一步融合C的特征
# 注意，这里用Conv2d而不是线性层，因此N个节点，每个的卷积核的值都是一样的，减少训练参数！
class output_layer(nn.Module):

    def __init__(self, args, input_length):
        super(output_layer, self).__init__()
        nhid = args.nhid
        # 输入特征(T-8)*64，即4*64，输出特征固定128
        self.fully_1 = torch.nn.Conv2d(input_length * nhid, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        # 输入特征固定128，输出特征固定1
        self.fully_2 = torch.nn.Conv2d(128, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, data):
        # (B, T', N, C)
        _, time_num, node_num, feature_num = data.size()
        data = data.permute(0, 2, 1, 3)  # (B, T', N, C)->(B, N, T', C)  # 时间维T'=4后移
        data = data.reshape([-1, node_num, time_num*feature_num, 1])  # (B, N, T', C)->(B, N, T'*C, 1)  # 与特征C'=64合并，另创建维度，这里T'*C=256
        data = data.permute(0, 2, 1, 3)  # (B, N, T'*C, 1)->(B, T'*C, N, 1)  # 时间维前移恢复
        data = self.fully_1(data)  # (B, 128, N, 1)  # Conv2d，仅改变特征T'*C -> 128
        data = torch.relu(data)  # 过滤负值
        data = self.fully_2(data)  # (B, 1, N, 1)  # Conv2d，仅改变特征128 -> 1
        data = data.squeeze(dim=3)  # (B, 1, N)  # 消除最后1维
        return data  # (B, 1, N)


class stsgcn(nn.Module):

    def __init__(self, args, A):
        super(stsgcn, self).__init__()
        self.A = A
        num_of_vertices = args.num_nodes  # 170 - 节点数量
        self.layer_num = args.layer_num  # 4 - 串联的STSGCL层数
        input_length = args.seq_length  # 12 - 输入时间步
        input_features_num = args.nhid  # 64
        num_of_features = args.nhid  # 64
        self.predict_length = args.num_for_predict  # 12 - 要预测的时间步
        # self.mask = nn.Parameter(torch.where(A>0.0,1.0,0.0).cuda(), requires_grad=True).cuda()
        # self.mask = nn.Parameter(torch.ones(3*num_of_vertices, 3*num_of_vertices).cuda(), requires_grad=True).cuda()
        self.mask = nn.Parameter(torch.rand(3*num_of_vertices, 3*num_of_vertices).to(args.device), requires_grad=True).to(args.device)
        self.stsgcl = nn.ModuleList()
        for _ in range(self.layer_num):  # 创建4个STSGCL（串联）
            self.stsgcl.append(stsgcl(args, input_length, input_features_num))  # 输入序列=12，特征=64
            input_length -= 2  # 下一个STSGCL输入序列长度减2
            input_features_num = num_of_features  # 节点特征数：第一层是C，往后都是C'，这里C=C'=64
        self.output_layer = nn.ModuleList()  # 创建12个输出层
        for _ in range(self.predict_length):
            self.output_layer.append(output_layer(args, input_length))  # 每个输入序列长度=4
        # 输入层 Conv2d，特征1->64
        self.input_layer = torch.nn.Conv2d(args.in_dim, args.nhid, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, input):
        # (B, T, N, C)  (32, 12, 170, 1)
        input = input.permute(0, 3, 2, 1)  # 转置成(B, C, N, T) (32, 1, 170, 12)
        data = self.input_layer(input)  # 二维卷积，特征维C提升至64：(32, 64, 170, 12)
        data = torch.relu(data)  # 过滤负数，提取非线性特征
        data = data.permute(0, 3, 2, 1)  # 恢复成(B,T,N,C') (32, 12, 170, 64)
        adj = self.mask * self.A  # 随机矩阵mask 乘 邻接矩阵A 哈达玛积，均为[3N, 3N]
        for i in range(self.layer_num):  # x输入到4个串联的STSGCL中
            data = self.stsgcl[i](data, adj)  # 上一个输出是下一个输入！！！
        # (B, T, N, C') -> (B, T-2, N, C') -> (B, T-4, N, C') -> (B, T-6, N, C') -> (B, T-8, N, C')
        need_concat = []
        for i in range(self.predict_length):  # 循环12次，同一个data，传入12个不同的输出层
            output = self.output_layer[i](data)  # x(B, T-8, N, C')，输出层浓缩T-8、融合C'特征，返回(B, 1, N)
            need_concat.append(output.squeeze(1))  # 消去中间1的维度 (B, N)
        outputs = torch.stack(need_concat, dim=1)  # (B, 12, N) 堆叠12个输出，作为对12个时间步的预测
        outputs = outputs.unsqueeze(3)  # (B, 12, N, 1)  # 在末尾追加1的维度
        return outputs





