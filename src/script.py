import argparse
from NsgaII import Nsga_II
from utils.load_data import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NAS argument parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fmnist', 'cifar10', 'cifar100'], help='set the dataset to be used', required=True)
    parser.add_argument('-n', '--name', help='name of the folder created for results', required=True)
    parser.add_argument('-g', '--n_gen', help='number of generations', type=int, required=True)
    parser.add_argument('-p', '--pop_size', help='number of individual in the starting population', type=int, required=True)
    parser.add_argument('-rc', '--rate_crossover', help='how often crossover might occur (between 0 and 1)', default=0.7, required=False, type=float)
    parser.add_argument('-rm', '--rate_mutation', help='how often mutation might occur (between 0 and 1)', default=0.1, required=False, type=float)
    parser.add_argument('-rl', '--rate_local_search', help='how often local_search might occur (between 0 and 1)', default=0.5, required=False, type=float)
    parser.add_argument('-s', '--step_size', help='size of the step for local search', default=0.1, required=False, type=float)

    args = parser.parse_args()
    config = vars(args)

    if args.dataset == "mnist":
        train_data, test_data = load_mnist(size=(2000, 500))
    elif args.dataset == "fmnist":
        train_data, test_data = load_fashion_mnist()
    elif args.dataset == "cifar10":
        train_data, test_data = load_cifar_10()
    elif args.dataset == "cifar100":
        train_data, test_data = load_cifar_100()

    config.pop('dataset')
    config['train_data'] = train_data
    config['test_data'] = test_data


    nas = Nsga_II(**config)
    if nas.valid:
        nas.optimise()
