import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from 毕业设计.SocialGCN.parser import parse_args

args= parse_args()

class Data(object):
    def __init__(self, path, batch_size,model_type):
        self.path = path  # ‘../Data/yelp_’
        self.batch_size = batch_size  # 1024
        self.model_type = model_type
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        social_file = path + '/adj_matrix_social.npz'

        # get social matrix
        self.social_mat = sp.load_npz(social_file)
        # get train_friends
        self.train_friends = {}
        social_mat_coo =self.social_mat.tocoo()
        for i,j in zip(social_mat_coo.row,social_mat_coo.col):
            if i not in self.train_friends:
                self.train_friends[i] = [j]
            else:
                self.train_friends[i].append(j)

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}  # 记录每个用户的负样本

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0]) # 第一列是用户id
                    self.exist_users.append(uid) # 添加到用户集中
                    self.n_items = max(self.n_items, max(items)) # 记录物品数量，必须记录最大的id，因为要转换为矩阵
                    self.n_users = max(self.n_users, uid) # 同上
                    self.n_train += len(items)  # 记录训练集中 交互 的数量

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        # 因为是从0开始的，所以要加上1
        self.n_items += 1
        self.n_users += 1
        # 打印n_users,n_items等上面这些数据
        self.print_statistics()
        # 论文的R，即user-item邻接矩
        # 阵;dok_matrix是存储稀疏矩阵的一种方式，字典存，key是位置，value是值，这里只是初始化，后面会赋值
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) # 用户-物品交互矩阵

        self.train_items, self.test_set = {}, {}

        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:] # 用户即其交互过的物品
                    for i in train_items:
                        # 邻接矩阵对应值置1
                        self.R[uid, i] = 1.
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        alpha = args.alpha
        # SocialGCN: 加入用户-用户社交关系
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32) # (M+N)x(M+N)
        adj_mat = adj_mat.tolil()
        # 转换为lil_matrix格式，使用lil.data和lil.rows两个列表的列表存数据，data存每行中的非零元素，rows存这些非零元素所在的列
        R = self.R.tolil()
        # 论文中的R’
        R_ = R.multiply(alpha)
        adj_mat[:self.n_users, self.n_users:] = R_
        adj_mat[self.n_users:, :self.n_users] = R_.T
        # 加入社交关系，论文中的S'
        adj_mat_social = self.social_mat.tolil()
        adj_mat[:self.n_users,:self.n_users] = adj_mat_social.multiply(1-alpha)
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        t2 = time()

        def mean_adj_single(adj):
            #  return D^-1 * A
            # adj.sum(1)计算每行的和，然后转换成np.array格式
            rowsum = np.array(adj.sum(1))
            # 下面是计算D
            d_inv = np.power(rowsum, -1).flatten() # rowsum即每个节点的度(包括自己)
            d_inv[np.isinf(d_inv)] = 0. # 0^-1为inf，要重新置为0
            d_mat_inv = sp.diags(d_inv) # D矩阵

            mean_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return mean_adj.tocoo()

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0])) # D^-1/2AD^-1/2=D^-1(A+I)
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys(): # u是uid
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u])) # 用户u没有交互过的物品
            pools = [rd.choice(neg_items) for _ in range(100)] # 对用户u负采样100次
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self): # 抽样，抽一个batch出来进行train
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size) # 用户数量足够，抽batch_size个用户出来，无放回抽样
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)] # 用户数量不够，有放回抽样

        def sample_pos_items_for_u(u, num): # 为用户u进行正抽样，抽num个，即sample交互过的物品
            # sample num pos items for u-th user
            pos_items = self.train_items[u] # pos_items是正物品列表，注意是从训练样本中取且训练样本中全为正
            n_pos_items = len(pos_items) # 正物品数量
            pos_batch = [] # 存抽的样
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num): # 抽负样本
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items: # 不在训练样本中的即为负样本
                    neg_items.append(neg_id)
            return neg_items

        def sample_pos_friends_for_u(u, num):
            # sample num pos friends for u-th user
            if u not in self.train_friends: # 没有好友
                random_friend = np.random.randint(low=0,high=self.n_users,size=1)[0]
                return [random_friend]
            pos_friends = self.train_friends[u] # pos_friends为该用户所有的好友列表
            n_pos_friends = len(pos_friends) # 好友数量
            pos_batch = [] # 存抽的样
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_friends, size=1)[0]
                pos_i_id = pos_friends[pos_id]
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_friends_for_u(u, num):
            # sample num neg friends for u-th user
            neg_friends = []
            while True:
                if len(neg_friends) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_users,size=1)[0]
                if  u not in self.train_friends or neg_id not in self.train_friends[u] and neg_id not in neg_friends: # 不在训练样本中的即为负样本
                    neg_friends.append(neg_id)
            return neg_friends

        pos_items, neg_items, pos_friends, neg_friends = [], [], [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            pos_friends+=sample_pos_friends_for_u(u,1)
            neg_friends+=sample_neg_friends_for_u(u,1)

        # 每个用户一个正样本一个负样本
        return users, pos_items, neg_items, pos_friends, neg_friends

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        """
        这是论文后面的sparsity测试，即交互过的物品越多的用户，推荐的性能越好，所以要对用户进行稀疏性分类

        """
        all_users_to_test = list(self.test_set.keys()) # 测试用户集合
        user_n_iid = dict() # key为交互过的物品数量，value为用户id，比如{20,[1,2,3]}表示交互的物品数量为20的用户有1，2，3

        # generate a dictionary to store (key=n_iids, value=a list of uid). iid为item_id
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.分别为交互过的物品数量<24,<50<117<1014
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
