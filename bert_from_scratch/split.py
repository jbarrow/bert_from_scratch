import pandas as pd
import random

from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    data = pd.read_csv(args.dataset, header=None, names=['line', 'company_id', 'header', 'clause'])

    eval_digit = dev_digit = random.randint(0, 9)

    while eval_digit == dev_digit:
        eval_digit = random.randint(0, 9)

    fp_train = open('train.txt', 'w')
    fp_dev = open(f'dev_{dev_digit}.txt', 'w')
    fp_eval = open(f'eval_{eval_digit}.txt', 'w')

    for row in tqdm(data.itertuples()):
        fp = fp_train
        if (row.company_id % 10) == dev_digit:
            fp = fp_dev
        elif (row.company_id % 10) == eval_digit:
            fp = fp_eval

        if type(row.header) is str:
            fp.write(row.header)
            fp.write('\n')
        if type(row.clause) is str:
            fp.write(row.clause)
            fp.write('\n')
        fp.write('\n')
