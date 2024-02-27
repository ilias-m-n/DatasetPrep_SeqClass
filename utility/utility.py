from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
import pandas as pd
import os
import re
from tqdm import tqdm
from itertools import product
import math

def round_half_up(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier

def round_half_away_from_zero(n, decimals=0):
    rounded_abs = round_half_up(abs(n), decimals)
    return math.copysign(rounded_abs, n)

def create_stratified_split(df, label = "label", split_sizes = {"training": .8, "validation": .1, "test": .1}, micro_labels = None, seed = 10):

    if (micro_labels == None) or (len(micro_labels) == 0):
        return create_stratified_split_no_micro(df, split_sizes=split_sizes)

    res_df = {}
    for s in split_sizes:
        res_df[s] = pd.DataFrame()

    unique_values_per_micro_col = [df[micro].unique() for micro in micro_labels]
    
    cart_prod_micro = list(product(*unique_values_per_micro_col))

    for item in cart_prod_micro:
        micro_pair_mask = True
        for i, ml in enumerate(micro_labels):
            micro_pair_mask &= (df[ml] == item[i])

        inter_df = create_stratified_split_no_micro(df[micro_pair_mask], label, split_sizes, seed)
        for s in split_sizes:
            res_df[s] = pd.concat([res_df[s], inter_df[s]])

    for s in split_sizes:
        res_df[s].reset_index(drop=True, inplace=True)
   
    return res_df

def create_stratified_split_no_micro(df, label = "label", split_sizes = {"train": .8, "val": .1, "test": .1}, seed = 10):

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
            indexes[l][s] = [int(round_half_up(current_index)), int(round_half_up(current_index+no_samples * split_sizes[s], 0))]
            current_index += no_samples * split_sizes[s]
                    
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
        try:
            with open(file_path, "r") as file:
                res = file.read()
        except UnicodeDecodeError:
            return None
        else:
            # checks whether text contains words temporary solution
            # data should be cleaned before creating datasets
            pattern = re.compile(r'\w+')
            if not bool(re.search(pattern, res)):
                return None
    return res

def create_dataset_dict(data_split, features):
    for s in data_split:
        data_split[s] = Dataset.from_pandas(data_split[s], features)

    return DatasetDict(data_split)

def load_data(from_hub = True, dataset_name_hub = "", path_dataset = ""):
    """
    Loads a DatasetDict from either a local directory or the HuggingFace Hub.

    Args:
        from_hub (Boolean): Determines whether to load dataset from Hub or a local directory.
        dataset_name_hub (String/List(String)): Name of Dataset to be loaded from Hub.
                                        If String: Corresponding dataset will be loaded.
                                        If List: Corresponding datasets will be loaded and merged.
        path_dataset (String/List(String)): Filepath to local dataset.
                                        If String: Corresponding dataset will be loaded.
                                        If List: Corresponding datasets will be loaded and merged.

    Returns:
        DatasetDict
    """
    if from_hub:
        if isinstance(dataset_name_hub, list):
            mult_datasets = []
            # load individual datasets
            for ds in dataset_name_hub:
                mult_datasets.append(load_dataset(ds))
            # concatenate datasets
            if len(mult_datasets) == 1:
                return mult_datasets[0]
            return merge_datasetdicts(mult_datasets)
        else:
            return load_dataset(dataset_name_hub)
    else:
        if isinstance(path_dataset, list):
            mult_datasets = []
            #load individual datasets
            for ds in path_dataset:
                mult_datasets.append(load_from_disk(ds))
            # concatenate datasets
            if len(mult_datasets) == 1:
                return mult_datasets[0]
            return merge_datasetdicts(mult_datasets)
        else:
            return load_from_disk(path_dataset)

def merge_datasetdicts(list_dsd):
    """
    Merges DatasetDicts along their splits.

    Args:
        list_dsd (List(DatasetDicts)): List containing DatasetDict objects.

    Returns:
        Merged DatasetDict.
    """
    datasets = {}

    for split in list_dsd[0]:
        datasets[split] = list_dsd[0][split]

    for i in range(1,len(list_dsd)):
        for split in datasets:
            datasets[split] = concatenate_datasets([datasets[split], list_dsd[i][split]])
    
    return DatasetDict(datasets)

def det_num_spec_tokens(tokenizer):

    special_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens)
    test_sentence_token_ids = tokenizer("Hello World!")["input_ids"] 
    occurence = 0

    for token in test_sentence_token_ids:
        if token in special_token_ids:
            occurence += 1
    
    return occurence

def apply_sliding_window_to_datasetdict(datasetdict, col_id, col_text, tokenizer, max_token_per_example, special_token_ids, 
                                         slide_step_size, col_label = None, no_windows = {"front":1, "back":1}, flag_full = False):
    result_df_dict = {}

    for split in datasetdict:
        print(split)
        result_df_dict[split] = apply_sliding_window_to_dataset(datasetdict[split], col_id, col_text, tokenizer, max_token_per_example, special_token_ids, 
                                                            slide_step_size, col_label, no_windows, flag_full)

    return result_df_dict    


def apply_sliding_window_to_dataset(dataset, col_id, col_text, tokenizer, max_token_per_example, special_token_ids, 
                                    slide_step_size, col_label = None, no_windows = {"front":1, "back":1}, flag_full = False):
    
    result_df = pd.DataFrame()
    
    for i in tqdm(range(int(dataset.num_rows))):
        temp = apply_sliding_window_to_example(dataset[i], col_id, col_text, tokenizer, max_token_per_example, 
                                               special_token_ids, slide_step_size, col_label, no_windows, flag_full)
        result_df = pd.concat([result_df, temp])
    result_df.reset_index(drop="True", inplace=True)

    return result_df

def apply_sliding_window_to_example(ex_dict, col_id, col_text, tokenizer, max_token_per_example, special_token_ids, 
                                    slide_step_size, col_label = None, no_windows = {"front":1, "back":1}, flag_full = False):

    id = ex_dict[col_id]
    tokens = tokenizer(ex_dict[col_text])["input_ids"]
    # remove special tokens
    tokens = [token for token in tokens if token not in special_token_ids]
    num_tokens = len(tokens)

    num_segments = max(0,math.ceil((num_tokens-max_token_per_example)/slide_step_size)) + 1
    indeces = [(slide_step_size*i, max_token_per_example + slide_step_size * i) for i in range(num_segments)]
    indeces[-1] = (indeces[-1][0], num_tokens +1)

    if not flag_full:
        num_segments = sum([no_windows[side] for side in no_windows])
        front_indeces = [indeces[i] for i in range(no_windows["front"])]
        back_indeces = [indeces[i] for i in range(-1,-no_windows["back"]-1,-1)]
        indeces = front_indeces + back_indeces

    ids = [id + "_" + str(i) for i in range(num_segments)]
    segments = [tokenizer.decode(tokens[index[0]:index[1]]) for index in indeces]

    result = pd.DataFrame({"id":ids, "filename":[id]*num_segments, "text":segments})
    if col_label:
        result[col_label] = ex_dict[col_label]

    return result