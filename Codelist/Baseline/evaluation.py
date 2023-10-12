import numpy as np
import torch
from utils import get_feature
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score
from torch import nn
from class_labels import *
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
    map      : float (MAP)
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


def inference(model, test_loader, criterion, device):
    softmax = nn.Softmax(dim=1)
    model = model.to(device)
    model.eval()
    label_list, predict_list = torch.tensor([], device=device), torch.tensor([], device=device)
    total_loss = 0.0
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            id, wave_data, labels, duration = batch
            labels = labels.to(device)
            for i in range(len(id)):
                new_data = torch.tensor([], device=device)
                voice_seg_list = speech_seg(wave_data[i], duration, sr=22050, max_len=5, min_len=3, overlap=1)
                n = len(voice_seg_list)
                new_labels = labels[i].repeat(n)
                for x in voice_seg_list:
                    feat = get_feature(x.numpy(), sr=22500, frame_len=1024, win_step=1 / 4, n_mels=256)
                    new_data = torch.cat([new_data, torch.as_tensor(feat).unsqueeze(0).to(device)], dim=0)
                outputs = model(new_data)
                batch_loss = criterion(outputs, new_labels)
                total_loss += batch_loss.item()
                outputs = softmax(outputs).mean(0)
                predict_list = torch.cat((predict_list, outputs.unsqueeze(0)), 0)

            labels = torch.zeros(outputs.unsqueeze(0).size(), device=device).scatter_(1, labels.unsqueeze(1).data, 1)
            label_list = torch.cat((label_list, labels), 0)

    total_loss = total_loss / len(test_loader)

    predict_list = np.asarray(predict_list.to('cpu'))
    label_list = np.asarray(label_list.to('cpu'))
    map = mean_average_precision(predict_list, label_list)
    mrr = MRR(predict_list, label_list)
    top_1 = top_n(predict_list, label_list, 1)
    top_5 = top_n(predict_list, label_list, 5)
    return mrr * 100, top_1 * 100, top_5 * 100, map * 100, total_loss, label_list, predict_list



def transfer_to_hierary(labels, hier_class_list):
    class_list, genus_list, family_list, order_list = hier_class_list
    hier_labels = []
    for i in labels:
        spe = class_list[i].split(' ')[-1]
        hier_labels.append([i,
                            genus_list.index(HIERARY[spe][2]),
                            family_list.index(HIERARY[spe][1]),
                            order_list.index(HIERARY[spe][0])])
    return hier_labels

def Hier_dis_of_mis(label_list, predict_list, hier_class_list, transfer_to_hier=True):
    if transfer_to_hier:
        label_list = transfer_to_hierary(label_list, hier_class_list)
        predict_list = transfer_to_hierary(predict_list, hier_class_list)
    hier_acc = np.zeros(4)
    d = 0.0
    for i in range(len(predict_list)):
        for k in range(len(hier_acc)):
            if predict_list[i][k] == label_list[i][k]:
                hier_acc[k] += 1
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
    hier_acc = hier_acc / len(label_list) * 100
    macc = (hier_acc[0] + hier_acc[1] + hier_acc[2] + hier_acc[3]) / 4
    return d, hier_acc, macc

def confusion_matrix(label_list, predict_list, species_list, save_cm=True, save_cm_path=''):
    label_list = label_list.argmax(1)
    predict_list = predict_list.argmax(1)
    batch_corr = 0.0
    cm = np.zeros((len(species_list), len(species_list)), dtype=int)
    for i in range(len(label_list)):
        label_spe = species_list.index(SELECT_CLASS[label_list[i]])
        predict_spe = species_list.index(SELECT_CLASS[predict_list[i]])
        if label_spe == predict_list[i]:
            cm[label_spe][predict_spe] += 1
            batch_corr += 1
        else:
            cm[label_spe][predict_spe] += 1

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
            writer.writerow(['PRECISION']+presicion+[accuracy])
    return cm, accuracy, recall, presicion

if __name__ == '__main__':
    import sys
    sys.path.append('../../BirdCLEF/')
    import json
    from torch.utils.data import DataLoader
    from model import Xception
    from dataloader import BirdsoundData

    hier_class_list = [SELECT_CLASS, SELECT_GENUS, SELECT_FAMILY, SELECT_ORDER]
    split_datas = json.load(open(sys.path[-1] + 'SplitDatas/small_dataset1.json'))
    test_datas = split_datas['origin']['test']
    test_dataset = BirdsoundData(test_datas, option='test', class_list=hier_class_list[0])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, drop_last=True, pin_memory=True)

    model = Xception(num_classes=len(hier_class_list[0]), use_attention=True)
    checkpoint = torch.load(sys.path[-1] + 'Results/Baseline/GINN_woHier/ckpt/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss()

    mrr, top_1, top_5, map, loss, label_list, predict_list = inference(model, test_loader, criterion, device='cuda')
    HDM, hier_acc, macc = Hier_dis_of_mis(label_list.argmax(1), predict_list.argmax(1), hier_class_list, transfer_to_hier=True)
    print("Test results: loss = {:0.3f}, MRR = {:0.3f}%, MAP = {:0.3f}%, top1_acc = {:0.3f}%, top5_acc = {:0.3f}%" \
          .format(loss, mrr, map, top_1, top_5))
    print("corr_spe_acc = {:0.3f}%, fam_acc = {:0.3f}%, gen_acc = {:0.3f}%, ord_acc = {:0.3f}%, macc = {:0.3f}%, HDM = {:0.5f}" \
        .format(hier_acc[0], hier_acc[1], hier_acc[2], hier_acc[3], macc, HDM))

    cm, accuracy, recall, presicion = confusion_matrix(label_list, predict_list, SELECT_CLASS, save_cm=True,
                                                       save_cm_path=sys.path[
                                                                        -1] + "Results/Baseline/GINN_woHier/total.csv")
    """
    label_list = transfer_to_hierary(label_list.argmax(1), hier_class_list)
    predict_list = transfer_to_hierary(predict_list.argmax(1), hier_class_list)
    # Count the correct number by category
    spe_acc = np.zeros(len(hier_class_list[0]))
    for i in range(len(label_list)):
        if label_list[i][0] == predict_list[i][0]:
            spe_acc[label_list[i][0]] += 1
    gen_acc = np.zeros(len(hier_class_list[1]))
    for i in range(len(label_list)):
        if label_list[i][1] == predict_list[i][1]:
            gen_acc[label_list[i][1]] += 1
    fam_acc = np.zeros(len(hier_class_list[2]))
    for i in range(len(label_list)):
        if label_list[i][2] == predict_list[i][2]:
            fam_acc[label_list[i][2]] += 1
    ord_acc = np.zeros(len(hier_class_list[3]))
    for i in range(len(label_list)):
        if label_list[i][3] == predict_list[i][3]:
            ord_acc[label_list[i][3]] += 1

    spe_num = np.zeros(len(hier_class_list[0]))
    gen_num = np.zeros(len(hier_class_list[1]))
    fam_num = np.zeros(len(hier_class_list[2]))
    ord_num = np.zeros(len(hier_class_list[3]))
    for i in range(len(label_list)):
        ord_num[label_list[i][3]] += 1
        fam_num[label_list[i][2]] += 1
        gen_num[label_list[i][1]] += 1
        spe_num[label_list[i][0]] += 1
    """
