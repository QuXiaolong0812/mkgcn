import torch
from model import MKGCN
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def train(args, data_loader):
    num_user, num_item, num_entity, num_relation = data_loader.load_dataset_info()
    kg = data_loader.load_knowledge_graph()
    multimodal_dict = data_loader.load_modals_dict(args)
    user_history_dict = data_loader.load_user_history_dict()
    train_loader, test_loader, train_dataset, test_dataset = data_loader.load_dataset(args)

    device = torch.device(('cuda:'+str(args.gpu)) if torch.cuda.is_available() else 'cpu')
    print('model training on {}.'.format(device))

    mkgcn = MKGCN(args, num_user, num_entity, num_relation, kg, user_history_dict, multimodal_dict, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(mkgcn.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    # top-k evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_setting(args, train_dataset, test_dataset, num_item)

    for epoch in range(args.n_epochs):
        train_loss = 0.0
        train_auc = 0.0
        train_acc = 0.0
        train_f1 = 0.0

        for user_ids, item_ids, labels in train_loader:
            user_ids, item_ids, labels = user_ids.to( device ), item_ids.to( device ), labels.to( device )
            optimizer.zero_grad()
            scores = mkgcn(user_ids, item_ids)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_auc += roc_auc_score( labels.cpu().detach().numpy(), scores.cpu().detach().numpy() )
            scores_list = [1 if item > 0.5 else 0 for item in scores.cpu().detach().numpy()]
            train_acc += np.mean( np.equal( scores_list, labels.cpu().detach().numpy() ) )
            train_f1 += f1_score(labels.cpu().detach().numpy(), scores_list)

        # evaluate per every epoch
        with torch.no_grad():
            mkgcn.eval()
            test_loss = 0.0
            test_auc = 0.0
            test_acc = 0.0
            test_f1 = 0.0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                scores = mkgcn(user_ids, item_ids)
                test_loss += criterion(scores, labels).item()
                test_auc += roc_auc_score(labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
                scores_list = [1 if item > 0.5 else 0 for item in scores.cpu().detach().numpy()]
                test_acc += np.mean(np.equal(scores_list, labels.cpu().detach().numpy()))
                test_f1 += f1_score( labels.cpu().detach().numpy(), scores_list )
            mkgcn.train()

        print( 'epoch %d    train auc: %.4f  acc: %.4f  f1: %.4f    test auc: %.4f  acc: %.4f  f1: %.4f'
               % (epoch, train_auc / (len( train_loader ) + float( "1e-8" )),
                  train_acc / (len( train_loader ) + float( "1e-8" )),
                  train_f1 / (len( train_loader ) + float( "1e-8" )),
                  test_auc / (len( test_loader ) + float( "1e-8" )),
                  test_acc / (len( test_loader ) + float( "1e-8" )),
                  test_f1 / (len( test_loader ) + float( "1e-8" )),) )

        if args.show_loss:
            print( 'epoch %d    train loss: %.4f    test loss: %.4f' % (
                epoch, train_loss / (len( train_loader ) + float( "1e-8" )),
                test_loss / (len( test_loader ) + float( "1e-8" ))) )
        if args.show_topk:
            precision, recall, ndcg = topk_eval(mkgcn, user_list, train_record, test_record, item_set, k_list, args.batch_size, device)
            print('precision: ', end='')
            for i in precision:
                print('%.4f\t'%i, end='')
            print()
            print( 'recall: ', end='' )
            for i in recall:
                print( '%.4f\t' % i, end='' )
            print()
            print( 'ndcg: ', end='' )
            for i in ndcg:
                print( '%.4f\t' % i, end='' )
            print('\n')

def get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def topk_setting(args, train_dataset, test_dataset, num_item):
    if args.show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        # remember turn DataFrame to ndarray
        train_record = get_user_record(train_dataset.values, True)
        test_record = get_user_record(test_dataset.values, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(num_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5

def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size, device):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items = test_item_list[start:start+batch_size]
            scores = model(torch.LongTensor([user]*batch_size).to(device), torch.LongTensor(items).to(device))
            for item, score in zip(items, scores):
                item_score_map[item] = score.item()
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items = test_item_list[start:]
            padding = [test_item_list[-1]]*(batch_size-len(test_item_list)+start)
            items += padding
            scores = model( torch.LongTensor( [user] * batch_size ).to( device ),
                            torch.LongTensor( items ).to( device ) )
            for item, score in zip( items, scores ):
                item_score_map[item] = score.item()

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x:x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_list = list(set(item_sorted[:k])&test_record[user])
            hit_num = len(set(item_sorted[:k])&test_record[user])
            dcg = np.sum([1/np.log2(pos + 2) for pos in range(k) if item_sorted[pos] in hit_list])
            idcg = np.sum([1/np.log2(pos + 2) for pos in range(min(k, len(test_record[user])))])
            precision_list[k].append( hit_num / k )
            recall_list[k].append( hit_num / len( test_record[user] ) )
            ndcg_list[k].append(dcg/idcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg
