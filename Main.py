import argparse
import logging
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from data import load_data
from GPAResNet import GPAResNet
from trainer import Trainer
from Transformer_origin import Transformer
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    # you can change the config parameters here according to your task
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--savepath', type=str, default='./Results')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--datapath', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # parse agrs
    args = parse_args()
    logger.info(vars(args))

    # select device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s', args.device)

    # set seed
    set_seed(args.seed)

    # get data
    train_loader, test_loader = load_data(args)

    # get model
    model = GPAResNet(d_model=200, ndead=5, batch_first=True)
    
    # if you want to resume training, you can uncomment the following line. It is also necessary for parameter sharing in our model.
    # model.load_state_dict(torch.load('model.pth'))

    # This is the ablation model in the paper. If you want to utilize for other tasks, you can change the parameters according to your task.
    #model = Transformer(num_embeddings=50265, embedding_dim=200, pad_idx=1, nhead=4, num_layers=6, num_classes=2)
    model = torch.nn.DataParallel(model)
    model.to(args.device)

    trainer = Trainer(model, args)
    trainer.train(train_loader, test_loader)