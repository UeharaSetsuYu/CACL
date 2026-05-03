import itertools
from tqdm import tqdm
import torch
import time
import pandas as pd
from Training import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED = True
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Main_Leaning(args, dataset_name, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc, best_ari, best_nmi = 0, 0, 0
    data = collections.defaultdict(list)

    print("============Foundation Information==========")
    if torch.cuda.is_available():
        print("GPU: ", torch.cuda.get_device_name(0))
    else:
        print("CPU")
    print("Dataset:", dataset_name)
    print("Pretrain epochs: ", args.pre_train)
    print("Stage two training epochs: ", args.epochs - args.pre_train)
    print("n_critic: ", args.n_critic)
    print("batch_size: ", args.batch_size)
    print("seed: ", args.seed)
    if args.missing_rate == 0:
        print("Complete data")
    else:
        print('Incomplete data, and missing rate: ', args.missing_rate)
    print("============ Training Processing ===========")

    # Training
    result, loss = main_train(args, dataset_name, config, device)  # schedual learning rate

    if best_acc < result['bestacc']:
        data['bestAcc'].append(result['bestacc'])
        data['bestAri'].append(result['bestari'])
        data['bestNmi'].append(result['bestnmi'])
        data['best_precision'].append(result['best_precision'])
        data['bestpur'].append(result['bestpur'])
        data['bestfmea'].append(result['bestfmea'])

    return data

def main():

    dataset = { 
        1: "BDGP",                  # 2 views, it`s across view datas   
    }
    args = parse_args()
    dataset_name = dataset[args.datasets]  # 指明需要使用的数据集，这里的--dataset 中默认为0，即Caltech101-20数据集
    config = get_default_config(dataset_name)
    robust = {'acc': [], 'ari': [], 'nmi': [], 'pur': [] }
    seed = 5

    print("============ sensitivity ===========")
    for i in range(args.times):
        print(f"Seed: {seed}")
        args.seed = seed
        data = Main_Leaning(args, dataset_name, config)
        robust['acc'].append(np.max(data['bestAcc']))
        robust['ari'].append(np.max(data['bestAri']))
        robust['nmi'].append(np.max(data['bestNmi']))
        robust['pur'].append(np.max(data['bestpur']))
        seed += 5

    print(f"ACC: Mean-{np.mean(robust['acc']):.4f}, std-{np.std(robust['acc']):.4f},"
          f"ARI: Mean-{np.mean(robust['ari']):.4f}, std-{np.std(robust['ari']):.4f},"
          f"NMI: Mean-{np.mean(robust['nmi']):.4f}, std-{np.std(robust['nmi']):.4f},"
          f"PUR: Mean-{np.mean(robust['pur']):.4f}, std-{np.std(robust['pur']):.4f},")
    print(robust)

    # save result
    df = pd.DataFrame(robust)
    df.to_csv(dataset_name + '_incomplete_result.csv', index=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Elapsed Time: ' + str(time.time() - start_time))








