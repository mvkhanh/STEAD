import torch
import torch.nn.functional as F
import option
args=option.parse_args()
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve

torch.autograd.set_detect_anomaly(True)

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def distance(self, x, y):
        d = torch.cdist(x, y, p=2)
        return d

    def forward(self, feats, margin = 100.0):
        bs = len(feats)
        n_feats = feats[:bs // 2]
        a_feats = feats[bs // 2:]
        n_d = self.distance(n_feats, n_feats)
        a_d = self.distance(n_feats, a_feats)
        n_d_max, _ = torch.max(n_d, dim=0)
        a_d_min, _ = torch.min(a_d, dim=0)
        a_d_min = margin - a_d_min
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        a_d_min = torch.max(torch.zeros(bs // 2).to(device), a_d_min)
        return torch.mean(n_d_max) + torch.mean(a_d_min)

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.triplet = TripletLoss()

    def forward(self, scores, feats, targets, alpha = 0.01):
        loss_ce = self.criterion(scores, targets)
        loss_triplet = self.triplet(feats)
        return loss_ce, alpha * loss_triplet

def train(loader, model, optimizer, scheduler, device, epoch):

    with torch.set_grad_enabled(True):
        model.train()
        pred = []
        label = []
        loss_t = 0
        for step, (ninput, nlabel, ainput, alabel) in tqdm(enumerate(loader)):
            input = torch.cat((ninput, ainput), 0).to(device)
            
            scores, feats, = model(input) 
            pred += scores.cpu().detach().tolist()
            labels = torch.cat((nlabel, alabel), 0).to(device)
            label += labels.cpu().detach().tolist()

            loss_criterion = Loss()
            loss_ce, loss_con = loss_criterion(scores.squeeze(), feats, labels)
            loss = loss_ce + loss_con
            loss_t = loss
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step_update(epoch * len(loader) + step)
        fpr, tpr, _ = roc_curve(label, pred)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(label, pred)
        pr_auc = auc(recall, precision)
        print('train_pr_auc : ' + str(pr_auc))
        print('train_roc_auc : ' + str(roc_auc))
        print(loss_t)
        return  loss.item()
