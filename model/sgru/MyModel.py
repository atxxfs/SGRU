import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        # x=(B, N, 2H)  A=(N, N1)  -> (B, N1, 2H)
        x = torch.einsum('bnh,nm->bmh', (x, A))
        return x.contiguous()


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, category="Linear", bias=True, activate="ReLU", dropout_rate=0.0):
        super(Linear, self).__init__()
        if category == "Linear":
            self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        elif category == "Conv1d":
            self.linear = nn.Conv1d(in_dim, out_dim, kernel_size=(1,), bias=bias)
        elif category == "Conv2d":
            self.linear = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), bias=bias)
        else:
            raise ValueError("ERROR: Unsupported Linear Category!")
        self.activation = None
        if activate == "ReLU":
            self.activation = nn.ReLU()
        elif activate == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activate == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activate == "Tanh":
            self.activation = nn.Tanh()
        elif activate != "None":
            raise ValueError("ERROR: Unsupported Activation Function!")
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        return x


class PureGRU(nn.Module):

    def __init__(self, batch_size, num_node, embedding_dim, hidden_dim, device):
        super().__init__()
        self.device = torch.device(device)  # "cuda:0"
        self.batch_size = batch_size
        self.num_node = num_node
        self.embedding_dim = embedding_dim  # D'=10
        self.hidden_dim = hidden_dim
        self.nconv = NConv()
        self.hidden_forward_tensor = []  # len = t
        self.hidden_forward_tensor_at = 0  # [0, t-1]
        self.weight_setup = Linear(embedding_dim, hidden_dim)  # D -> H
        self.weight_reset = Linear(hidden_dim, hidden_dim, activate="Sigmoid")
        self.weight_update = Linear(hidden_dim, hidden_dim, activate="Sigmoid")
        self.weight_chebyshev_connect = Linear(4 * hidden_dim, hidden_dim)
        self.weight_hidden = Linear(2 * hidden_dim, hidden_dim, activate="Sigmoid")
        self.ht_1 = None
        self.initialize()

    def forward(self, xt, embedding_matrix_1, embedding_matrix_2):

        # (B, N, D) --> (B, N, H)
        xt = self.weight_setup(xt)
        # xt(B, N, H)  --cat--  ht_1(B, N, H)  ===  Xh(B, N, 2H)
        xh = torch.cat((xt, self.ht_1[:xt.shape[0], ...]), dim=2)

        # # GCN Part

        # Softmax(ReLU(emb * embT))  (N, N)
        A = F.softmax(F.relu(torch.mm(embedding_matrix_1, embedding_matrix_2.transpose(0, 1))))
        # Identity Matrix (N, N)
        E = torch.eye(self.num_node).to(self.device)
        # An = (B, N, 2H)
        A1 = self.nconv(xh, A)
        A2 = self.nconv(xh, E)
        xh = torch.cat((A1, A2), dim=2)
        # dim reduce (B, N, 8H) --> (B, N, H)
        xh = self.weight_chebyshev_connect(xh)
        # Reset/Update (B, N, H) --> (B, N, H)
        r = self.weight_reset(xh)
        z = self.weight_update(xh)
        # Hidden Fuse  (B, N, H)  --cat--  (B, N, H)  ===  (B, N, 2H)  --> (B, N, H)
        ht_hat = self.weight_hidden(torch.cat((xt, z * self.ht_1[:xt.shape[0], ...]), dim=2))
        # Balance output ht_hat and ht_1.  return (B, N, H)
        ht = r * self.ht_1[:xt.shape[0], ...] + (1 - r) * ht_hat
        return ht

    def initialize(self, specified_tensor=None):
        if specified_tensor is not None:
            self.ht_1 = specified_tensor
        if self.ht_1 is None:
            self.ht_1 = torch.randn(self.batch_size, self.num_node, self.hidden_dim).to(self.device)


class MyModel(nn.Module):

    def __init__(self, args):
        super(MyModel, self).__init__()
        self.batch_size = args.batch_size  # B=64
        self.num_node = args.num_nodes  # N=170
        self.input_dim = args.input_dim  # D=1
        self.input_length = args.lag  # T=12
        self.embedding_dim = 10  # D'=10
        self.hidden_dim = args.rnn_units  # 64
        self.output_length = args.horizon  # 3
        self.output_dim = args.output_dim  # D=1
        self.k_size = 3  # conv_kernel
        self.d_size = 1  # conv_dilation

        # (1, 1, N, D)
        self.s_embedding = nn.Parameter(torch.empty(1, 1, self.num_node, self.input_dim))
        # (1, T, 1, D)
        self.t_embedding = nn.Parameter(torch.empty(1, self.input_length, 1, self.input_dim))


        self.preparation = Linear(self.input_dim, self.embedding_dim, dropout_rate=0.0)  # 1 -> 10
        self.embedding_matrix_1 = nn.Parameter(torch.FloatTensor(self.num_node, self.embedding_dim))  # (N, D')
        self.embedding_matrix_2 = nn.Parameter(torch.FloatTensor(self.num_node, self.embedding_dim))  # (N, D')

        self.gru_1_1 = PureGRU(self.batch_size, self.num_node, self.embedding_dim, self.hidden_dim, args.device)
        self.gru_1_2 = PureGRU(self.batch_size, self.num_node, self.embedding_dim, self.hidden_dim, args.device)

        self.gte_2_1_a = Linear(self.hidden_dim, self.hidden_dim, activate="Sigmoid")
        self.gte_2_1_b = Linear(self.hidden_dim, self.hidden_dim, activate="None")
        self.gru_2_1 = PureGRU(self.batch_size, self.num_node, self.embedding_dim, self.hidden_dim, args.device)

        self.gte_2_2_a = Linear(self.hidden_dim, self.hidden_dim, activate="Sigmoid")
        self.gte_2_2_b = Linear(self.hidden_dim, self.hidden_dim, activate="None")
        self.gru_2_2 = PureGRU(self.batch_size, self.num_node, self.embedding_dim, self.hidden_dim, args.device)

        self.gte_2_3_a = Linear(self.hidden_dim, self.hidden_dim, activate="Sigmoid")
        self.gte_2_3_b = Linear(self.hidden_dim, self.hidden_dim, activate="None")
        self.gru_2_3 = PureGRU(self.batch_size, self.num_node, self.embedding_dim, self.hidden_dim, args.device)

        self.zipper = Linear(
            3 * self.input_length,
            self.output_length,
            category="Conv2d",
            dropout_rate=0.0,
            activate="LeakyReLU"
        )

        self.output = Linear(self.hidden_dim, self.output_dim, activate="LeakyReLU")
        self.params_initialize()

    def params_initialize(self):
        nn.init.kaiming_normal_(self.embedding_matrix_1)
        nn.init.kaiming_normal_(self.embedding_matrix_2)
        nn.init.xavier_uniform_(self.s_embedding)
        nn.init.xavier_uniform_(self.t_embedding)

    def forward(self, x):
        # x(B, T, N, D) -> (B, T, N, D')
        x = self.preparation(x)
        # emb
        x = x + self.t_embedding + self.s_embedding
        # GRU on T
        ht_gru_1_1 = None
        ht_gru_1_2 = None
        ht_list = []
        seq_length = x.shape[1]

        for t in range(seq_length):
            ht_gru_1_1 = self.gru_1_1(x[:, t, :, :], self.embedding_matrix_1, self.embedding_matrix_2)
        for t in range(seq_length):
            ht_gru_1_2 = self.gru_1_2(x[:, t, :, :], self.embedding_matrix_1, self.embedding_matrix_2)

        self.gru_2_1.initialize(self.gte_2_1_a(ht_gru_1_1) * self.gte_2_1_b(ht_gru_1_2))
        self.gru_2_2.initialize(self.gte_2_2_a(ht_gru_1_1) * self.gte_2_2_b(ht_gru_1_2))
        self.gru_2_3.initialize(self.gte_2_3_a(ht_gru_1_1) * self.gte_2_3_b(ht_gru_1_2))
        for t in range(seq_length):
            ht_gru_2_1 = self.gru_2_1(x[:, t, :, :], self.embedding_matrix_1, self.embedding_matrix_2)
            ht_list.append(ht_gru_2_1)
        for t in range(seq_length):
            ht_gru_2_2 = self.gru_2_2(x[:, t, :, :], self.embedding_matrix_1, self.embedding_matrix_2)
            ht_list.append(ht_gru_2_2)
        for t in range(seq_length):
            ht_gru_2_3 = self.gru_2_3(x[:, t, :, :], self.embedding_matrix_1, self.embedding_matrix_2)
            ht_list.append(ht_gru_2_3)

        # ht(B, N, H) --stack--> out(B, kT, N, H)
        out = torch.stack(ht_list, dim=1)
        # zipper --> (B, T', N, H)
        out = self.zipper(out)
        # Output -> (B, T', N, 1)
        out = self.output(out)
        # (B, T', N, 1)
        return out
