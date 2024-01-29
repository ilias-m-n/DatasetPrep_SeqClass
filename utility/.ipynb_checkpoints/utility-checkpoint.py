from collections import Counter
from datasets import Dataset, DatasetDict
import pandas as pd
import os
import re

def create_stratified_split(df, label = "label", split_sizes = {"training": .9, "validation": .1}, seed = 10):

    # return dfs
    res_dfs = {}

    # for each label we create a shuffled sub df and their respective data split indexes
    indexes = {}
    sub_dfs = {}
    labels = df[label].unique()
    for l in labels:
        # shuffle sub df
        sub_dfs[l] = df[df[label] == l].sample(frac=1, random_state=seed).copy()
        sub_dfs[l].reset_index(drop=True, inplace=True)
        # number of samples in sub df
        no_samples = sub_dfs[l].shape[0]
        # run index to determine indexes of splits
        current_index = 0
        # create indexes for data splits
        indexes[l]  = {}
        for s in split_sizes:
            indexes[l][s] = [current_index, int(round(current_index+no_samples * split_sizes[s], 0))]
            current_index += int(round(no_samples * split_sizes[s], 0))
        # correct indexes to full range
        if indexes[l][list(split_sizes.keys())[-1]][-1] < no_samples:
            for s in split_sizes:
                indexes[l][s][1] += 1
                if indexes[l][s][0] != 0:
                    indexes[l][s][0] += 1
                    
    # create data splits
    for s in split_sizes:
        res_dfs[s] = pd.DataFrame()
        for l in labels:
            res_dfs[s] = pd.concat([res_dfs[s], sub_dfs[l].iloc[indexes[l][s][0]:indexes[l][s][1],:].copy()])
        res_dfs[s].reset_index(drop=True, inplace=True)    

    return res_dfs

def parse_txt(file_path):
    res = None
    # first check whether file exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            res = file.read()
            # checks whether text contains words temporary solution
            # data should be cleaned before creating datasets
            pattern = re.compile(r'\w+')
            if not bool(re.search(pattern, res)):
                return None
    return res

def create_dataset_dict(data_split, cols):
    for s in data_split:
        data_split[s] = Dataset.from_pandas(data_split[s][cols])

    return DatasetDict(data_split)