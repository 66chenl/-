

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(SGCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size # 64
        self.batch_size = args.batch_size # 1024
        self.node_dropout = args.node_dropout[0] # 0.1
        self.mess_dropout = args.mess_dropout # [0.1,0.1,0.1]
        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size) # [64,64,64]
        self.decay = eval(args.regs)[0] # 1e-5
        self.alpha = args.alpha

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
    # 随机初始化user和item的embedding以及W和b
    def init_weight(self):
        # xavier init，初始化基于“为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等”
        initializer = nn.init.xavier_uniform_
        # Parameter继承于torch.Tensor,对一些辅助函数进行了定义；并且会自动添加到Module类的参数列表中
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,self.emb_size))), # n_user * emb_size
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,self.emb_size))) # n_item * emb_size
        })
        # 存weight matrix W
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers # [64,64,64,64] 第一个为emb_size;注意没有修改self.layers
        for k in range(len(self.layers)): # 总共循环3次
            # W_gc_k 存的是第k层的W1
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            # b_gc_K 存的是bias，文中没提
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict
    # 将scipy.sparse的稀疏矩阵转换为pytorch中的稀疏矩阵
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col]) # [[coo_row],[coo_col]]
        v = torch.from_numpy(coo.data).float() # 将numpy数据转换为tensor
        return torch.sparse.FloatTensor(i, v, coo.shape) # torch.sparse用于存稀疏矩阵，格式与coo一样，indices+data
    # 对节点进行dropout
    def sparse_dropout(self, x, rate, noise_shape):
        # noise_shape为稀疏矩阵中非0元素的数量
        random_tensor = 1 - rate
        # torch.rand(size)会产生于一个长为size,元素范围为[0,1)的列表；然后这里每个元素加上1-rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        # dropout_mask的格式为 [True,True,False,True...]
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        # 一些node被dropout
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        # 乘一个是因为部分元素被dropout了，剩下的元素加强训练
        return out * (1. / (1 - rate))
    # 损失函数
    def create_bpr_loss(self, users, pos_items, neg_items,pos_friends, neg_friends):
        # 传进来的是user_embedding, pos_item_embedding以及neg_item_embedding，...仍然是对应位置属于同一个用户
        pos_i_scores = torch.sum(torch.mul(users, pos_items), axis=1) # 结果仍是列表，每个用户的正物品得分
        neg_i_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        pos_f_scores = torch.sum(torch.mul(users,pos_friends), axis=1)
        neg_f_scores = torch.sum(torch.mul(users,neg_friends), axis=1)
        # 这个即文中定义的单个用户的loss
        rating_loss = self.alpha * nn.LogSigmoid()(pos_i_scores - neg_i_scores)
        social_loss = (1-self.alpha) * nn.LogSigmoid()(pos_f_scores-neg_f_scores)
        maxi =rating_loss + social_loss
        # 所有用户loss取平均
        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer；正则项
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, pos_friends, neg_friends, drop_flag=True):
        """

        Args:
            users: [用户1,用户2...]
            pos_items: [用户1对应的pos_item,用户2对应的pos_item...]
            neg_items: [用户1对应的neg_item,用户2对应的neg)item...]
            pos_friends:[用户1对应的pos_friend，用户2对应的pso_friend,...]
            neg_friends:[用户1对应的neg_friend,用户2对应的neg_friend,...]
        Returns:
            u_g_embeddings：[用户1的embedding,用户2的embedding,...]
            pos_i_g_embeddings: [用户1对应的pos_item的embedding，...]
            neg_i_g_embeddings: [用户1对应的neg_item的embedding,...]
            pos_f_g_embeddings: [用户1对应的pos_friend的embedding,...]
            neg_f_g_embeddings: [用户1对应的neg_friend的embedding,...]

        """
        # 对邻接矩阵进行dropout，即node dropout,torch.sparse.FloatTensor._nnz是稀疏矩阵中非0元素的数量
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            # size_embedding即 LE
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            # transformed sum messages of neighbors.即LEW1
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]
            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings)

            # message dropout;mess_dropout为[0.1,0.1,0.1]
            # nn.dropout(ratio)返回一个函数，将张量中的一些元素设置为0
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # 归一化，即将张量的每一个元素除以张量的模长
            # p=2表示张量的模长是所有元素的平方和的开方；dim是计算模长的维度，1表示每一行计算模长
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            # 将所有embedding存起来
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1) # 将所有embedding拼接
        u_g_embeddings_temp = all_embeddings[:self.n_user, :] # 这里是所有用户的embedding
        i_g_embeddings = all_embeddings[self.n_user:, :] # 这里是所有物品embedding

        """
        *********************************************************
        look up.
        """
        # 返回训练数据对应的用户、正样本、负样本的embedding
        u_g_embeddings = u_g_embeddings_temp[users, :]
        pos_f_g_embeddings = u_g_embeddings_temp[pos_friends, :]
        neg_f_g_embeddings = u_g_embeddings_temp[neg_friends, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]


        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, pos_f_g_embeddings, neg_f_g_embeddings
