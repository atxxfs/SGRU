import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class STID(nn.Module):
    """
    The implementation of CIKM 2022 short paper
        "Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting"
    Link: https://arxiv.org/abs/2208.05233
    """

    def __init__(self, args):
        super().__init__()
        # attributes
        self.num_nodes = args.num_nodes
        self.node_dim = args.node_dim
        self.input_len = args.input_len
        self.input_dim = args.input_dim
        self.embed_dim = args.embed_dim
        self.output_len = args.output_len
        self.num_layer = args.num_layer
        self.temp_dim_tid = args.temp_dim_tid
        self.temp_dim_diw = args.temp_dim_diw
        self.time_of_day_size = args.time_of_day_size
        self.day_of_week_size = args.day_of_week_size

        self.if_time_in_day = args.if_T_i_D
        self.if_day_in_week = args.if_D_i_W
        self.if_spatial = args.if_node

        # spatial embeddings
        if self.if_spatial:
            # (N, 32)
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            # (288, 32)
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            # (7, 32)
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        # 3L -> 32
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding  32 + (32 + 32 + 32)
        self.hidden_dim = self.embed_dim+self.node_dim * \
            int(self.if_spatial)+self.temp_dim_tid*int(self.if_time_in_day) + \
            self.temp_dim_diw*int(self.if_day_in_week)

        # 128 -> 128
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        # 128 -> 12
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of STID.
        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data  (B, L, N, 3)
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 0]  # (B, L, N)
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            # 在(288, 32)上取下标，t_i_d_data在0~1之间，因此乘288，再转换成整数，最终维度[B, 1, N]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 0]
            # 在(7, 32)上取下标，再转换成整数，最终维度[B, 1, N]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        # (B, N, L, 3) reshape成(1, B, 3L, N)
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # (1, B, 32, N)
        time_series_emb = self.time_series_emb_layer(input_data)  # (B, 32, N)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            # (N, 32) --unsqueeze--> (1, N, 32) --expand--> (B, N, 32) --transpose--> (B, 32, N)
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))  # (B, 32, N)
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))  # (B, 32, N)

        # concat all embeddings
        # 4个连接之后是  (B, 128, N)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding  (B, 128, N)
        hidden = self.encoder(hidden)

        # regression  (B, 12, N)
        prediction = self.regression_layer(hidden)

        return prediction
