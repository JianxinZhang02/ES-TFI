import argparse
import sys
import os
import warnings
sys.path.append(os.path.dirname(sys.path[0]))

warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ES-TFI example')
    parser.add_argument('--model', type=str, default='ES-TFI', help='use model', choices=['ES-TFI'])
    parser.add_argument('--dataset', type=str, default='qb_video', help='use dataset')
    parser.add_argument('--mutation', type=int, default=1, help='use mutation: 1 use 0 not used')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    dataset = parser.parse_args().dataset

    if dataset == 'kuairand':
        parser.add_argument('--label', type=str, default=['is_click', 'is_like'], help='label file name')
        from run.run_kuairand import train
    elif dataset == 'qb_video':
        parser.add_argument('--label', type=str, default=['click', 'like'], help='label file name')
        from run.run_qb_video import train
    elif dataset == 'ali_ccp':
        parser.add_argument('--label', type=str, default=['click', 'purchase'], help='label file name')
        from run.run_ali_ccp import train
    args = parser.parse_args()
    train(params=args)
