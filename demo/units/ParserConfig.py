import argparse


def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--datasets', type=int, default='0', help='dataset id')
    parser.add_argument('--epochs', type=int, default='700', help='number of epochs')
    parser.add_argument('--batch_size', type=int, default='256', help='batch size')
    parser.add_argument('--view_num', type=int, default=3, help='number of view')
    parser.add_argument('--train_rate', type=float, default=0.8, help='train data rate')
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('--lr', type=float, default=1.0e-4, help='learning rate')
    parser.add_argument('--pre_train', type=int, default=300, help='pre-train times')
    parser.add_argument('--n_critic', type=int, default=1, help='Discriminator Training numeration')
    parser.add_argument('--model', type=str, default='Clustering', help='Or Classification') 
    parser.add_argument('--epsilon', type=float, default=0.8, help='hyper-parameters')
    parser.add_argument('--beta', type=int, default=3, help='hyper-parameters')
    parser.add_argument('--times', type=int, default=1, help='training times')
    parser.add_argument('--eta', type=int, default=10, help='hyper-parameters')
    args = parser.parse_args()

    return args
 
