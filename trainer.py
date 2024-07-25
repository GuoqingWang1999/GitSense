import torch.optim as optim
import torch
import os
import pandas as pd
import logging
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import csv
logger = logging.getLogger(__name__)

# the metrics for evaluation (F1, precision, recall)
def all_metrics(y_true, y_pred, is_training=False):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1.item(), precision.item(), recall.item(), tp.item(), tn.item(), fp.item(), fn.item()

class Trainer():
    def __init__(self, model, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("./longformer-base-4096")
        self.model = model
        self.metrics = {'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0, 'mcc': 0, 'g_mean': 0}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        # define weights to balance the loss
        weights = torch.tensor([0.6, 0.4])
        self.criterion = torch.nn.CrossEntropyLoss(weights).to(args.device)

    def train(self, train_loader, dev_loader):
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch, train_loader)
            self.eval_epoch(epoch, dev_loader)
            logging.info('Epoch %d finished' % epoch)

    def savemodel(self, k):
        # if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
        #     os.mkdir(os.path.join(self.args.savepath, self.args.dataset))
        torch.save({'state_dict': self.model.state_dict(),
                    'k': k,
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.savepath,
                                f'model_{k}.pth'))
        logger.info(f'save:{k}.pth')

    def train_epoch(self, epoch, train_loader):
        self.model.train()

        loss_num = 0.0
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (inputs, features, label) in enumerate(pbar):
            # if you have no enough memory, you can reduce the max_length to a smaller number
            #token = self.tokenizer(list(inputs), padding=True, return_tensors='pt', max_length=1000, truncation=True)
            token = self.tokenizer(list(inputs), padding=True, return_tensors='pt', max_length=50000, truncation=True)
            ids = token['input_ids'].to(self.args.device)
            label = label.to(self.args.device)
            features = features.to(self.args.device)
            outputs = self.model(ids,features)
            loss = self.criterion(outputs, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_num += loss.item()
            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=loss.sum().item())

        self.savemodel(epoch)

    def eval_epoch(self, epoch, dev_loader):
        self.model.eval()

        all_labels = []
        all_preds = []
        pbar = tqdm(dev_loader, total=len(dev_loader))
        with torch.no_grad():
            for i, (inputs,features, label) in enumerate(pbar):
                # if you have no enough memory, you can reduce the max_length to a smaller number
                #token = self.tokenizer(list(inputs), padding=True, return_tensors='pt', max_length=1000, truncation=True)
                token = self.tokenizer(list(inputs), padding=True, return_tensors='pt', max_length=50000,truncation=True)
                ids = token['input_ids'].to(self.args.device)
                label = label.to(self.args.device)
                features = features.to(self.args.device)
                outputs = self.model(ids,features)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            # open a file to save the prediction results
            filename = f'./Results/pred_result_{epoch}.csv'
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['label', 'pred'])  
                for a, b in zip(all_labels, all_preds):
                    writer.writerow([a, b])

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
            self.update_best_scores(epoch, f1, precision, recall,tp, tn, fp, fn)

            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'
                .format(f1, precision, recall))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))

    def update_best_scores(self, epoch, f1, precision, recall, tp, tn, fp, fn):
        if f1 > self.metrics['f1'] or precision > self.metrics['precision'] or recall > self.metrics['recall']:
            self.metrics['f1'] = f1
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.scores2file(epoch, f1, precision, recall, tp, tn, fp, fn)

    def scores2file(self, epoch, f1, precision, recall, tp, tn, fp, fn,):
        save_path = self.args.savepath + '/result_record.csv'
        # add tp tn fp fn to self.matrix type dict
        _record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "epoch": epoch,
            "args": self.args
        }
        result_df = pd.DataFrame(_record, index=[0])
        result_df.to_csv(save_path, mode='a', index=False, header=True)