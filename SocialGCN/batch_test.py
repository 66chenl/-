'''
Created on 2023/2/9
@author: chenliao
'''

import metrics
from 毕业设计.SocialGCN.parser import parse_args
from load_data import *
import multiprocessing # 多线程处理
import heapq

cores = multiprocessing.cpu_count() // 2 # 用一半的core来运算

args = parse_args()
Ks = args.Ks # [20, 40, 60, 80, 100]

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, model_type=args.model_type)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    # user_pos_test表示确实交互过的物品，test_items表示测试物品列表，rating是每个测试物品的得分
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    # 利用堆排序返回测试得分最高的前K个物品
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            # 推荐的前K个里面确实交互过的就为1
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    """

    Args:
        user_pos_test: 测试用户真实购买过的
        r: 为大小为K的列表[1，1,0,1,...,1] 为1表示推荐的物品确实被交互过，表示推荐准了
        auc: area under curve
        Ks: [20,40,...]  推荐列表大小

    Returns: {'precision':[precision@20,], 'recall':[recall@20,...]}

    """

    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks: # 分别计算不同K值时的指标
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x): # 实际上是为一个batch的用户进行测试
    # user u's ratings for user u
    rating = x[0] # 评分矩阵 1 * item_batch
    u = x[1] # uid
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set，即测试用户交互过的物品
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))
    # 测试用户进行测试的物品
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    # result用来存test的各种指标
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2 # 128
    i_batch_size = BATCH_SIZE # 64
    # 测试用户列表
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs): # 测试用户一共要用多少个batch，直接全部丢到模型可能太复杂了
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end] # 0-127,切片不包含最后一个

        if batch_test_flag:
            # batch-item test，每次只用一部分物品
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
                # 注意，这些item并不一定是被测试用户交互过的or没交互过的
                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    # model只是输出user和item的embedding
                    # user_batch是测试用户列表，item_batch是正物品列表，没有负物品
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    # 记录测试的用户对一些物品的评分
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch # 测试结果记录在rate列表中
                i_count += i_rate_batch.shape[1] # 记录已经测过的物品数量

            assert i_count == ITEM_NUM

        else:
            # all-item test；直接对所有物品进行test
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,pos_f_g_embeddings, neg_f_g_embeddings\
                    = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              [],
                                                              [],
                                                              drop_flag=False)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,pos_f_g_embeddings, neg_f_g_embeddings\
                    = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              [],
                                                              [],
                                                              drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        # zip用户将多个迭代对象组合在一起，这里就是[(用户1的评分矩阵，用户1)，(用户2的评分矩阵，用户2)]
        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch) # rate_batch.numpy将矩阵转换成[[],[]]的形式
        # 表示在多个进程中并行地运行test_one_user函数,返回值是一个列表，每个进程的返回值
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result
