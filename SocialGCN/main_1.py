'''
Created on 2023/2/8

@author: Chenliao
'''
import os
import datetime
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from SGCN import SGCN
from helper import *
from batch_test import *

if __name__ == '__main__':
    # args来自batch_test
    args.device = torch.device('cuda')
    args.lr = 0.0001
    # 获取3种类型的矩阵
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    # eval函数是将字符串转换为数字
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    # 要给我们的模型传入用户数，物品数以及norm_adj邻接矩阵
    model = SGCN(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time()
    # cur_best_pre记录精确率
    cur_best_pre_0, stopping_step = 0, 0
    # Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 这些列表用来记录各种指标的变化，loss,precision,recall,ndcg,hitratio
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    """
    *********************************************************
    Train.
    """
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        # 训练数目/batch_size = 要训练的总次数
        n_batch = data_generator.n_train // args.batch_size + 1 # 加1因为最后剩下了一批数量不够batch_size的
        for idx in range(n_batch):

            users, pos_items, neg_items, pos_friends, neg_friends = data_generator.sample() # 产生训练数据
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, pos_f_g_embeddings, neg_f_g_embeddings= model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           pos_friends,
                                                                           neg_friends,
                                                                           drop_flag=args.node_dropout_flag)
            # 这里对u_g_embedding进行拼接

            # batch_loss即得分loss＋正则项，后面分别为得分loss以及正则项(embedding loss)
            # loss里面可以加入用户-用户项误差

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings,
                                                                              pos_f_g_embeddings,
                                                                              neg_f_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0: # verbose>0代表输出日志
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()

        """
        每第10次进行测试一下
        Test.
        """
        users_to_test = list(data_generator.test_set.keys()) # 测试用户列表
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss.item())
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            min_at = args.Ks[0]
            max_at = args.Ks[-1]
            perf_str = f'Epoch {epoch} [{t2-t1}s + {t3-t2}s]: train=[{loss}={mf_loss} + {emb_loss}];' \
                       f'recall@{min_at}=[{ret["recall"][0]}],recall@{max_at}=[{ret["recall"][-1]}];' \
                       f'precision@{min_at}=[{ret["precision"][0]}], precision@{max_at}=[{ret["precision"][-1]}];' \
                       f'hit@{min_at}=[{ret["hit_ratio"][0]}],hit@{max_at}=[{ret["hit_ratio"][-1]}];' \
                       f'ndcg@{min_at}=[{ret["ndcg"][0]}],ndcg@{max_at}=[{ret["ndcg"][-1]}]'
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    # recs: [[epoch10的recall@10,recall@20...],[epoch20的recall@10...],[],...]
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 ((idx+1)*10, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    # 记录这次训练过程
    date = str(datetime.date.today()).replace('-','_')
    time = str(datetime.datetime.now().hour)
    file_name = f'train_log_{date}_{time}'
    path = f'./train_log/{file_name}_lr0001'
    os.mkdir(path)

    # recs_: [ [epoch9的recall@10,epoch19的recall@10], [epoch9的recall@20,epoch19的recall@20] ]
    recs_ = []
    pres_ = []
    ndcgs_ = []
    hit_ = []
    loss_ = loss_loger
    for idx in range(len(args.Ks)):
        recs_.append(recs[:,idx])
        pres_.append(pres[:,idx])
        ndcgs_.append(ndcgs[:,idx])
        hit_.append(hit[:,idx])
    # 画图
    x = [(i+1)*10 for i in range(len(recs_[0]))] #所有图的x都是这个
    ## loss figure
    plt.suptitle('Loss')
    plt.scatter(x,loss_)
    plt.savefig(f'{path}/loss.png')
    plt.clf()
    ## recall figure
    plt.suptitle('Recall')
    for i  in range(len(recs_)):
        at = args.Ks[i]
        plt.plot(x,recs_[i],label=f'recall@{at}')
    plt.legend()
    plt.savefig(f'{path}/recall.png')
    plt.clf()
    ## precision figure
    plt.suptitle('Precision')
    for i in range(len(pres_)):
        at = args.Ks[i]
        plt.plot(x, pres_[i], label=f'precision@{at}')
    plt.legend()
    plt.savefig(f'{path}/precision.png')
    plt.clf()
    ## NDCG figure
    plt.suptitle('NDCG')
    for i in range(len(ndcgs_)):
        at = args.Ks[i]
        plt.plot(x, ndcgs_[i], label=f'NDCG@{at}')
    plt.legend()
    plt.savefig(f'{path}/ndcg.png')
    plt.clf()
    ## hit ration figure
    plt.suptitle('Hit Ratio')
    for i in range(len(hit_)):
        at = args.Ks[i]
        plt.plot(x, hit_[i], label=f'HR@{at}')
    plt.legend()
    plt.savefig(f'{path}/hitratio.png')
    # 记录loss以及各种指标
    with open(f'{path}/log.txt','w') as f:
        # 第一行是epoch
        f.write('epoch: ')
        for epoch in x:
            f.write(f'{epoch} ')
        f.write('\n')
        # 记录recall
        for idx,recall in enumerate(recs_):
            f.write(f'recall@{args.Ks[idx]}: ')
            for rec in recall:
                f.write(f'{rec} ')
            f.write('\n')
        # 记录precision
        for idx,precision in enumerate(pres_):
            f.write(f'precision@{args.Ks[idx]}: ')
            for pre in precision:
                f.write(f'{pre} ')
            f.write('\n')
        # 记录ndcg
        for idx, ndcg in enumerate(ndcgs_):
            f.write(f'NDCG@{args.Ks[idx]}: ')
            for ndc in ndcg:
                f.write(f'{ndc} ')
            f.write('\n')
        # 记录hit ration
        for idx, hr in enumerate(hit_):
            f.write(f'HR@{args.Ks[idx]}: ')
            for h in hr:
                f.write(f'{h} ')
            f.write('\n')

    # 保存模型
    torch.save(model.state_dict(), path + '/model.pkl')






