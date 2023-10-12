import numpy as np
from utils import get_feature
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score
from class_labels import *
from config import *
import torch

def speech_seg(wave_data, duration, sr=22050, max_len=5, min_len=3, overlap=1):
    """
    Segmentation rule: Each record is split into segments through a sliding window at five-second intervals,
    with a one-second overlap between consecutive segments. If the test audio is less than three seconds, a process of self-concatenation is carried out.
    :param voice_path: Input audio signal
    :param sr: The default sampling rate is 22.05kHz
    :param max_len: Maximum segment length
    :param min_len: Minimum segment length
    :param overlap: Overlapping time length
    :return: voice_seg_list, list
    """

    voice_seg_list = []

    while duration < min_len:
        wave_data = torch.cat([wave_data, wave_data], dim=-1)
        duration = duration * 2

    start_index = 0
    end_index = start_index + max_len
    while start_index + max_len <= duration:
        seg_i = wave_data[start_index * sr: end_index * sr]
        voice_seg_list.append(seg_i)
        start_index = end_index - overlap
        end_index = start_index + max_len
    res = wave_data[start_index * sr:]
    if duration-start_index >= min_len:  # save more than 3s
        voice_seg_list.append(res)
    return voice_seg_list

def top_n(prediction, target, n):
    score = 0
    for y_t, y_s in zip(target, prediction):
        top = np.argsort(y_s)[::-1]
        y = np.argmax(y_t)
        if y in top[:n]:
            score += 1
    return score/len(prediction)

def area_under_roc_curve(prediction, target):
    """
    y_trues  : [nb_samples, nb_classes]
    y_scores : [nb_samples, nb_classes]
    map      : float (AUROC)
    """
    auroc = roc_auc_score(target, prediction)
    return auroc

def mean_average_precision(prediction, target):
    """
    target  : [nb_samples, nb_classes]
    prediction : [nb_samples, nb_classes]
    map     : float (MAP)
    """
    aps = []
    for y_t, y_s in zip(target, prediction):
        ap = average_precision_score(y_t, y_s)
        aps.append(ap)
    return np.mean(np.array(aps))

def MRR(prediction, target):
    # Calculate the label ranking average precision (LRAP) for every sample
    """ e.g.
    target = np.array([[1, 0, 0], [0, 0, 1]])
    prediction = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    """
    return label_ranking_average_precision_score(target, prediction)


def inference(model, test_loader, criterion, hierarchical_classes, device):
    model = model.to(device)
    model.eval()

    label_list, predict_list = [], []
    for i in range(len(hierarchical_classes)):
        label_list.append(torch.tensor([], device=device))
        predict_list.append(torch.tensor([], device=device))

    total_loss = 0.0
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            id, wave_data, labels, duration = batch
            labels = labels.to(device)
            for i in range(len(id)):
                new_data = torch.tensor([], device=device)
                new_labels = torch.tensor([], dtype=torch.int64, device=device)
                voice_seg_list = speech_seg(wave_data[i], duration, sr=SAMPLE_RATE, max_len=5, min_len=3, overlap=1)
                for x in voice_seg_list:
                    feat = get_feature(x.numpy(), sr=SAMPLE_RATE, frame_len=FRAME_LEN, win_step=1 / 4, n_mels=N_MELS)
                    new_data = torch.cat([new_data, torch.as_tensor(feat).unsqueeze(0).to(device)], dim=0)
                    new_labels = torch.cat([new_labels, labels], dim=0)
                multih_fg_map, multih_fmatrixs, outputs, multih_att, multih_atts = model(new_data)
                batch_loss = criterion(outputs, new_labels)
                total_loss += batch_loss.item()
                for k, out in enumerate(outputs):
                    out = out.mean(0)
                    predict_list[k] = torch.cat((predict_list[k], out.unsqueeze(0)), 0)
                    label_list[k] = torch.cat((label_list[k],
                                               torch.zeros(out.unsqueeze(0).size(), device=device)
                                               .scatter_(1, labels[:, k].unsqueeze(0).data, 1)), 0)
    total_loss = total_loss / len(test_loader)

    # result_list
    map, mrr, top_1, top_5 = [], [], [], []
    for i in range(len(hierarchical_classes)):
        predict_list[i] = np.asarray(predict_list[i].to('cpu'))
        label_list[i] = np.asarray(label_list[i].to('cpu'))
        map.append(mean_average_precision(predict_list[i], label_list[i]) * 100)
        mrr.append(MRR(predict_list[i], label_list[i]) * 100)
        top_1.append(top_n(predict_list[i], label_list[i], 1) * 100)
        top_5.append(top_n(predict_list[i], label_list[i], 5) * 100)
    return mrr, top_1, top_5, map, total_loss, label_list, predict_list

def transfer_to_hierary(labels, hier_class_list):
    class_list, genus_list, family_list, order_list = hier_class_list
    hier_labels = []
    for i in labels:
        spe = class_list[i].split(' ')[-1]
        hier_labels.append([i,
                            genus_list.index(HIERARY[spe][2]),
                            family_list.index(HIERARY[spe][1]),
                            order_list(HIERARY[spe][0])])
    return hier_labels

from path_corr import path_correction2
def path_correction(label_list, predict_list, path_corr=True):
    """ e.g.
        input_hier = [150, 122, 42, 14]
        predict_score,(num_spe)
        class_list = [SELECT_CLASS, SELECT_GENUS, SELECT_FAMILY, SELECT_ORDER]
    """
    if path_corr:
        corr_path = numpy.empty((len(predict_list[0]), len(predict_list)), dtype=int)
        for i in range(len(predict_list[0])):
            predict_score = [k[i, :] for k in predict_list]
            input_hier = [np.expand_dims(k, 0).argmax(1)[0] for k in predict_score]
            # 正确路径 tree[input_hier[0]] == input_hier
            if tree[input_hier[0]] != input_hier:
                input_hier = path_correction2(input_hier, predict_score)
            input_hier = np.asarray(input_hier)
            corr_path[i, :] = input_hier
        for i in range(len(label_list)):
            label_list[i] = list(label_list[i].argmax(1))
        label_list = np.array(label_list).T
    else:
        for i in range(len(label_list)):
            label_list[i] = list(label_list[i].argmax(1))
            predict_list[i] = list(predict_list[i].argmax(1))
        label_list = np.array(label_list).T
        corr_path = np.array(predict_list).T

    return label_list, corr_path


def Hier_dis_of_mis(label_list, predict_list):
    batch_corr = torch.zeros(4)
    d = 0.0
    for i in range(len(predict_list)):
        for k in range(len(batch_corr)):
            if predict_list[i][k] == label_list[i][k]:
                batch_corr[k] += 1
        if predict_list[i][0] != label_list[i][0]:
            if predict_list[i][1] == label_list[i][1]:
                d += 1 / 4
            elif predict_list[i][2] == label_list[i][2]:
                d += 2 / 4
            elif predict_list[i][3] == label_list[i][3]:
                d += 3 / 4
            else:
                d += 1
    d /= len(label_list)
    batch_corr = batch_corr/len(label_list)*100
    macc = (batch_corr[0] + batch_corr[1] + batch_corr[2] + batch_corr[3])/4
    return d, batch_corr, macc

def confusion_matrix(label_list, predict_list, species_list, save_cm=True, save_cm_path=''):
    batch_corr = torch.zeros(4)
    cm = np.zeros((len(species_list), len(species_list)), dtype=int)
    for i in range(len(label_list)):
        label_spe = species_list.index(SELECT_CLASS[label_list[i][0]])
        predict_spe = species_list.index(SELECT_CLASS[predict_list[i][0]])
        if label_spe == predict_list[i][0]:
            cm[label_spe][predict_spe] += 1
            batch_corr[0] += 1
        else:
            cm[label_spe][predict_spe] += 1
        if label_list[i][1] == predict_list[i][1]:
            batch_corr[1] += 1
        if label_list[i][2] == predict_list[i][2]:
            batch_corr[2] += 1

    accuracy = batch_corr / len(predict_list) * 100
    recall = (np.diagonal(cm) / cm.sum(axis=1)).tolist()
    presicion = (np.diagonal(cm) / cm.sum(axis=0)).tolist()

    if save_cm:
        import csv
        with open(save_cm_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['CLASS_ID']+species_list+['ACC', 'TOT', 'RECALL'])
            for i in range(len(species_list)):
                writer.writerow([species_list[i]]+list(cm[i])+[cm[i][i], sum(cm[i]), recall[i]])
            writer.writerow(['PRECISION']+presicion+[accuracy[0]])
    return cm, accuracy, recall, presicion

if __name__ == '__main__':
    import sys
    sys.path.append('../../BirdCLEF/')
    import json
    from torch.utils.data import DataLoader
    from model_mod import CHRF
    from dataloader import BirdsoundData
    from loss import *
    hier_class_list = [SELECT_CLASS, SELECT_GENUS, SELECT_FAMILY, SELECT_ORDER]
    hierarchy = {'class': len(hier_class_list[0]), 'genus': len(hier_class_list[1]), 'family': len(hier_class_list[2]),
                 'order': len(hier_class_list[3])}
    split_datas = json.load(open(sys.path[-1] + 'SplitDatas/small_dataset1_with_hier.json'))
    test_datas = split_datas['origin']['test']
    test_dataset = BirdsoundData(test_datas, option='test', class_list=hier_class_list)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, drop_last=True, pin_memory=True)

    model = CHRF(hierarchy=hierarchy, use_attention=True)
    checkpoint = torch.load(sys.path[-1] + 'Results/Hierarchy with DataAug/GINN_aug/ckpt/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = HierLoss(hierarchy)

    mrr, top_1, top_5, map, loss, label_list, predict_list = inference(model, test_loader, criterion, list(hierarchy.keys()), device='cuda')
    label_list, predict_list = path_correction(label_list, predict_list, path_corr=True)

    HDM, batch_corr, macc = Hier_dis_of_mis(label_list, predict_list)
    print("Test results: loss = {:0.3f}, MRR = {:0.3f}%/{:0.3f}%/{:0.3f}%/{:0.3f}%, MAP = {:0.3f}%/{:0.3f}%/{:0.3f}%/{:0.3f}%, "
          "top1_acc = {:0.3f}%/{:0.3f}%/{:0.3f}%/{:0.3f}%, top5_acc = {:0.3f}%/{:0.3f}%/{:0.3f}%/{:0.3f}%" \
          .format(loss, mrr[0], mrr[1], mrr[2], mrr[3], map[0], map[1], map[2], map[3],
                         top_1[0], top_1[1], top_1[2], top_1[3], top_5[0], top_5[1], top_5[2], top_5[3]))

    print("corr_top1_acc = {:0.3f}%/{:0.3f}%/{:0.3f}%/{:0.3f}%, macc = {:0.3f}%, HDM = {:0.5f}" \
        .format(batch_corr[0], batch_corr[1], batch_corr[2], batch_corr[3], macc, HDM))

    cm, accuracy, recall, presicion = confusion_matrix(label_list, predict_list, SELECT_CLASS, save_cm=True,
                                                       save_cm_path=sys.path[
                                                                        -1] + "Results/Hierarchy with DataAug/GINN_aug/total.csv")
