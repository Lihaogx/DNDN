import datetime
import numpy as np
import os
import os.path as osp
import pickle
import random
import time
import torch
import wandb
import yaml

from scipy.io import savemat
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm

from model.DNDN import DNDN

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def load_dataset(name='REDDIT-BINARY', filt='dowker'):
    save_name = osp.join(dataset_path, 'GCB_{}_{}.pkl'.format(name, filt))
    with open(save_name, 'rb') as f:
        dict_save = pickle.load(f)
    return dict_save

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def train():
    model.train()
    Total_loss_0 = 0
    Total_loss_WD0 = 0
    Total_loss_WD1 = 0
    cnt_sample = 0

    PD_result = dict()
    label_list = []

    optimizer.zero_grad()
    for sample in tqdm(train_sample, desc='training'):
        data = dict_save[dataset_size][sample]
        if len(data) <= 2:
            continue

        PD = torch.FloatTensor(data['barcode']).cuda()
        filt_value = torch.FloatTensor(data['filtration_val']).cuda().view(-1, 1)
        source_edge_index = torch.LongTensor(data['source_edge_index']).cuda()
        sink_edge_index = torch.LongTensor(data['sink_edge_index']).cuda()

        source_edge_index = remove_self_loops(source_edge_index)[0]
        sink_edge_index = remove_self_loops(sink_edge_index)[0]

        x0, loss, loss_WD0, loss_WD1 = model(filt_value, source_edge_index, sink_edge_index, PD, p=p, type='train')

        Total_loss_0 += loss.cpu().detach()
        Total_loss_WD0 += loss_WD0.cpu().detach()
        Total_loss_WD1 += loss_WD1.cpu().detach()
        cnt_sample += 1

        PD_result['PD{}'.format(cnt_sample)] = x0.cpu().detach().numpy()
        label_list.append(dict_save[dataset_size][sample]['label'])

        loss.backward()
        if cnt_sample % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    PD_result['label'] = np.array(label_list)
    if save_train_result:
        savemat(osp.join(result_path, '{}_{}_PD_train.mat'.format(dataset, log_name)), PD_result)

    return Total_loss_0 / cnt_sample, Total_loss_WD0 / cnt_sample, Total_loss_WD1 / cnt_sample


def test():
    model.eval()
    Total_loss_0 = 0
    Total_loss_WD0 = 0
    Total_loss_WD1 = 0
    Total_loss_PI0 = 0
    Total_loss_PI = 0
    cnt_sample = 0

    for sample in tqdm(test_sample, desc='testing'):
        with torch.no_grad():
            data = dict_save[dataset_size][sample]
            if len(data) <= 2:
                continue

            PD = torch.FloatTensor(data['barcode']).cuda()
            PI = torch.FloatTensor(data['PI']).cuda()
            filt_value = torch.FloatTensor(data['filtration_val']).cuda().view(-1, 1)
            source_edge_index = torch.LongTensor(data['source_edge_index']).cuda()
            sink_edge_index = torch.LongTensor(data['sink_edge_index']).cuda()

            source_edge_index = remove_self_loops(source_edge_index)[0]
            sink_edge_index = remove_self_loops(sink_edge_index)[0]

            x0, loss, loss_WD0, loss_WD1, loss_PI0, emb_PI = model(filt_value, source_edge_index, sink_edge_index, PD, p=p, type='test')

            loss_PI = PILoss(emb_PI, PI)
            Total_loss_PI0 += loss_PI0.cpu().detach()

            Total_loss_0 += loss.cpu().detach()
            Total_loss_WD0 += loss_WD0.cpu().detach()
            Total_loss_WD1 += loss_WD1.cpu().detach()
            Total_loss_PI += loss_PI.cpu().detach()
            cnt_sample += 1

    return Total_loss_0 / cnt_sample, Total_loss_WD0 / cnt_sample, Total_loss_WD1 / cnt_sample, \
           Total_loss_PI0 / cnt_sample, Total_loss_PI / cnt_sample


# 测试所用时间
def stat_time(method):
    model.eval()
    Total_loss_0 = 0
    Total_loss_WD0 = 0
    Total_loss_WD1 = 0
    Total_loss_PI = 0
    cnt_sample = 0

    PD_result = dict()
    label_list = []

    t0 = time.time()
    for sample in tqdm(dict_save['big_graph'].keys(), desc='testing'):
        with torch.no_grad():
            data = dict_save['big_graph'][sample]
            PD = torch.FloatTensor(data['barcode']).cuda()
            PI = torch.FloatTensor(data['PI']).cuda()

            filt_value = torch.FloatTensor(data['filtration_val']).cuda().view(-1, 1)
            source_edge_index = torch.LongTensor(data['source_edge_index']).cuda()
            sink_edge_index = torch.LongTensor(data['sink_edge_index']).cuda()

            source_edge_index = remove_self_loops(source_edge_index)[0]
            sink_edge_index = remove_self_loops(sink_edge_index)[0]

            x0, loss, loss_WD0, loss_WD1, emb_PI = model(filt_value, source_edge_index, sink_edge_index, PD, p=p, type='test')

            loss_PI = PILoss(emb_PI, PI)
            Total_loss_0 += loss.cpu().detach()
            Total_loss_WD0 += loss_WD0.cpu().detach()
            Total_loss_WD1 += loss_WD1.cpu().detach()
            Total_loss_PI += loss_PI.cpu().detach()
            cnt_sample += 1

            PD_result['PD{}'.format(cnt_sample)] = x0.cpu().detach().numpy()
            label_list.append(dict_save['big_graph'][sample]['label'])
    t1 = time.time()

    print("Data: {}, Method: {}, Sample: {}, PD Loss: {}, WD0 Loss: {}, WD1 Loss: {}, PI loss: {}, Average time: {}s"
          .format(dataset, method, cnt_sample, Total_loss_0 / cnt_sample, Total_loss_WD0 / cnt_sample,
                  Total_loss_WD1 / cnt_sample, Total_loss_PI / cnt_sample, (t1-t0) / cnt_sample))

    if save_test_result:
        PD_result['label'] = np.array(label_list)
        savemat(osp.join(result_path, '{}_{}_PD_test.mat'.format(dataset, method)), PD_result)


def shuffle_data_in_order(data):
    data_num = len(data)
    assert data_num > 10

    new_data = []
    for i in range(5):
        new_data += data[int(0.1*data_num*i):int(0.1*data_num*(i+1))]
        new_data += data[int(0.1*data_num*(i+5)):int(0.1*data_num*(i+6))]

    return new_data


def check_paths(*paths):
    for path in paths:
        if not osp.exists(path):
            print('Warning: path {} doesn''t exist, auto make dirs for it.'.format(path))
            os.makedirs(path)


def init_wandb_config(config):
    config.dataset = dataset
    config.modification = log_name
    config.load_model = load_model
    if load_model:
        config.load_model_name = load_model_name

    config.epochs = epochs
    config.model_type = model_type
    config.fusion = fusion
    config.hidden_dim = hidden_dim
    config.num_layers = num_layers
    config.learning_rate = learning_rate
    config.weight_decay = weight_decay
    config.dropout = dropout
    config.batch_size = batch_size


if __name__ == "__main__":
    
    config = load_config('config.yml')
    
    dataset_path = config['dataset_path']
    model_path = config['model_path']
    result_path = config['result_path']

    dataset = config['dataset']
    
    log_name = config['log_name']

    load_model_name = config['load_model_name']
    load_model = config['load_model']
    
    log_WD = config['log_WD']
    use_wandb = config['use_wandb']
    dataset_size = config['dataset_size']
    save_train_result = config['save_train_result']
    save_test_result = config['save_test_result']
    train_set_ratio = config['train_set_ratio']

    epochs = config['epochs']
    tolerance = config['tolerance']
    model_type = config['model_type']
    fusion = config['fusion']
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    dropout = config['dropout']
    batch_size = config['batch_size']

    test_epoch_interval = config['test_epoch_interval']

    seed = config['seed']
    new_node_feat = config['new_node_feat']
    use_edge_attn = config['use_edge_attn']
    save_model = config['save_model']

    p = 2  # p-wasserstein distance
    kernel = 'wasserstein'  # loss for PDs

    check_paths(dataset_path, model_path, result_path)

    model = DNDN(in_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, out_dim=4,
                       new_node_feat=new_node_feat, use_edge_attn=use_edge_attn, combine=fusion).cuda()
    if load_model:
        model.load_state_dict(torch.load(osp.join(model_path, load_model_name)))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    PILoss = torch.nn.MSELoss()

    print('dataset: {}, method: {}'.format(dataset, log_name))

    dict_save = load_dataset(dataset)
    
    if use_wandb:
        wandb.login(key='your_wandb_key')
        wandb_run = wandb.init(entity=' ', project=dataset, name=log_name)
        init_wandb_config(wandb_run.config)

    sample = [k for k in dict_save[dataset_size].keys()]
    # random.seed(seed)
    # random.shuffle(sample)
    sample = shuffle_data_in_order(sample)

    train_sample = sample[:int(len(sample)*train_set_ratio)]
    test_sample = sample[int(len(sample)*train_set_ratio):]

    best_epoch = -1
    min_PD_loss = float('inf')
    min_WD0_loss = float('inf')
    min_WD1_loss = float('inf')
    min_PI0_loss = float('inf')
    min_PI_loss = float('inf')

    for epoch in range(1, epochs+1):
        train_loss, WD0_loss, WD1_loss = train()

        data_info = "Model: {}, Dataset: {}, Epoch: {}\n".format(model_type, dataset, epoch)
        train_info = "Training Loss: {}, WD0 loss: {}, WD1 loss: {}\n".format(train_loss, WD0_loss, WD1_loss)
        print(data_info, end='')
        print(train_info, end='')

        if use_wandb:
            if log_WD:
                wandb.log({"Train/PD Loss": train_loss,
                            "Train/WD0 Loss": WD0_loss,
                            "Train/WD1 Loss": WD1_loss}, step=epoch)
            else:
                wandb.log({"Train/PD Loss": train_loss}, step=epoch)

        log_info = "{}: \n".format(str(datetime.datetime.now()))
        log_info += data_info + train_info

        if epoch % test_epoch_interval == 0:
            PD_loss, WD0_loss, WD1_loss, PI0_loss, PI_loss = test()

            test_info = "Test Loss: {}, WD0 loss: {}, WD1 loss: {}, PI0 loss: {}, PI loss: {}\n".format(
                PD_loss, WD0_loss, WD1_loss, PI0_loss, PI_loss)
            print(test_info, end='')
            log_info += test_info

            if use_wandb:
                wandb.log({"Test/PD Loss": PD_loss,
                            "Test/WD0 Loss": WD0_loss,
                            "Test/WD1 Loss": WD1_loss,
                            "Test/PI0 Loss": PI0_loss,
                            "Test/PI Loss": PI_loss}, step=epoch)

            if PD_loss < min_PD_loss:
                best_epoch = epoch
                min_PD_loss = PD_loss
                min_WD0_loss = WD0_loss
                min_WD1_loss = WD1_loss
                min_PI0_loss = PI0_loss
                min_PI_loss = PI_loss
                if save_model:
                    torch.save(model.state_dict(),
                                osp.join(model_path, '{}_{}_{}_epoch{}.pt'.format(dataset, model_type, log_name, epoch)))
            elif epoch - best_epoch > tolerance:
                print("Early stop at epoch: {}".format(epoch))
                break

        with open(osp.join(result_path, "{}_{}_{}.txt".format(dataset, model_type, log_name)), "a") as f:
            if epoch % test_epoch_interval == 0:
                best_info = "Best epoch: {}, best loss: {}\n".format(best_epoch, min_PD_loss)
                log_info += best_info
                print(best_info)

                if use_wandb:
                    wandb.log({"Best/epoch": best_epoch,
                                "Best/PD Loss": min_PD_loss,
                                "Best/WD0 Loss": min_WD0_loss,
                                "Best/WD1 Loss": min_WD1_loss,
                                "Best/PI0 Loss": min_PI0_loss,
                                "Best/PI Loss": min_PI_loss}, step=epoch)

            f.write(log_info)

    if use_wandb:
        wandb_run.finish()
