import argparse
import csv
import json
import os
import os.path as osp
import random
import sys
from functools import partial
from multiprocessing import Pool
sys.path.append("../../")
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split


file_abs_path = os.path.split(os.path.realpath(__file__))[0]

HEADER = ['uid', 'pid', 'domain', 'label']


def write_header(domain_save_path):
    train_data_path, val_data_path, test_data_path = [osp.join(domain_save_path, path) for path in
                                                      ['train.csv', 'val.csv', 'test.csv']]
    with open(train_data_path, "w", encoding='utf8', newline='') as train, open(val_data_path, "w", encoding='utf8',
                                                                                newline='') as val, open(
        test_data_path, "w", encoding='utf8', newline='') as test:
        train_writer, val_writer, test_writer = csv.writer(train), csv.writer(val), csv.writer(test)
        [writer.writerow(HEADER) for writer in [train_writer, val_writer, test_writer]]

def file_exist(domain_save_path):
    for path in ['train.csv', 'val.csv', 'test.csv']:
        if not osp.exists(osp.join(domain_save_path, path)):
            return False
    return True


def sample_negative_data(subgoup, pid_range, ctr_ratio, domain):
    uid = subgoup[0]
    g = subgoup[1]
    negative_num = int(len(g['pid']) / ctr_ratio)
    exclude_item = g['pid'].unique().tolist()  # Excludes the items clicked by user
    sample_set = [pid for pid in pid_range if pid not in exclude_item]

    negative_data = pd.DataFrame(columns=['uid', 'pid', 'label'])
    try:
        if negative_num > len(sample_set):
            negative_sample_set = sample_set
        else:
            negative_sample_set = random.sample(sample_set, negative_num)

        negative_data['pid'] = negative_sample_set
        negative_data['uid'] = uid
        negative_data['domain'] = domain
        negative_data['label'] = 0
        # negative_train_data = [(uid, negative_pid, 0) for negative_pid in negative_sample_set]
        return negative_data
    except:
        print(exclude_item)
        print(sample_set)
        print(negative_num)
        return negative_data

def save_train_val_test(domain_save_path, df, conf):
    train_data_path, val_data_path, test_data_path = [osp.join(domain_save_path, path) for path in
                                                      ['train.csv', 'val.csv', 'test.csv']]
    df_train, df_val, df_test = split_stratified_into_train_val_test(df,
                                                                     stratify_colname='label',
                                                                     frac_train=conf['train_val_test'][0],
                                                                     frac_val=conf['train_val_test'][1],
                                                                     frac_test=conf['train_val_test'][2],
                                                                     random_state=conf['seed'])
    df_train.to_csv(train_data_path, index=False)
    df_val.to_csv(val_data_path, index=False)
    df_test.to_csv(test_data_path, index=False)

def split_stratified_into_train_val_test(df_input, stratify_colname='label',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    if len(df_temp) > 1:
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                          y_temp,
                                                          stratify=y_temp,
                                                          test_size=relative_frac_test,
                                                          random_state=random_state)
    else:
        df_test = df_temp
        df_val = df_temp.drop(index=df_temp.index)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def split_by_domain(processed_data, split_save_path, conf):
    domain_field = conf["domain_field"]
    domain_list = sorted(list(set(processed_data[domain_field])))
    domain_data_all = processed_data.rename(columns={"UserID": "uid", "MovieID": "pid", domain_field: "domain"}, inplace=False)
    domain_data_all = domain_data_all[['uid', 'pid', 'domain', 'label']]
    domain_data_all = domain_data_all.sample(frac=1.0)
    for n_domain in domain_list:
        domain_save_path = osp.join(split_save_path, "domain_{}".format(n_domain))
        if not file_exist(domain_save_path):
            if not osp.exists(domain_save_path):
                os.makedirs(domain_save_path)

            # Write header
            write_header(domain_save_path)

        damain_data = domain_data_all[domain_data_all["domain"] == n_domain]

        save_train_val_test(domain_save_path, damain_data, conf)
        with open(osp.join(domain_save_path, "domain_property.json"), "w") as f:
            json.dump({
                "domain_name": n_domain,
                "n_uid": max(damain_data["uid"])+1,
                "n_pid": max(damain_data["pid"])+1,
                "ctr_ratio": sum(damain_data["label"])/len(damain_data["label"]),
                "pid_range": list(set(damain_data["pid"]))
            }, f)

def preprocess_data(raw_data_path,  processed_data_path):
    # user_file = "raw_data/ml_1m/users.dat"
    user_file = osp.join(raw_data_path, "users.dat")
    user_data = pd.read_csv(user_file, sep="::", header=None,
                            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
    user_data = user_data[["UserID", "Gender", "Age", "Occupation"]]
    gender_map = {"F": 0, "M": 1}
    user_data["Gender"] = user_data["Gender"].map(gender_map)
    age_map = {1: 0, 18: 2, 25: 3, 35: 4, 45: 5, 50: 6, 56: 7}
    user_data["Age"] = user_data["Age"].map(age_map)
    # ratings_file = "raw_data/ml_1m/ratings.dat"
    ratings_file = osp.join(raw_data_path, "ratings.dat")
    ratings_data = pd.read_csv(ratings_file, sep="::", header=None, names=["UserID", "MovieID", "Rating", "Timestamp"])
    ratings_data["label"] = ratings_data["Rating"] / 5.0
    ratings_data["label"] = ratings_data["label"].apply(lambda x: 1.0 if x >= 1.0 else 0.0)
    ratings_data = ratings_data[["UserID", "MovieID", "label"]]
    user_data['UserID'] = user_data['UserID'].astype('int32')
    ratings_data['UserID'] = ratings_data['UserID'].astype('int32')
    rating_user_data = pd.merge(user_data, ratings_data, how="inner", on="UserID")
    if not file_exist(processed_data_path):
        if not osp.exists(processed_data_path):
            os.makedirs(processed_data_path)
    uid2id_path = osp.join(processed_data_path, "uid2id.json")
    with open(uid2id_path, "w") as f:
        json.dump({
            "id": max(rating_user_data["UserID"])+1
        }, f)
    pid2id_path = osp.join(processed_data_path, "pid2id.json")
    with open(pid2id_path, "w") as f:
        json.dump({
            "id": max(rating_user_data["MovieID"])+1
        }, f)

    return rating_user_data


def split_to_domains(conf):
    raw_data_path = osp.join(file_abs_path, conf['raw_data_path'])
    split_save_path = osp.join(file_abs_path, conf['split_save_path'])
    processed_data_path = osp.join(split_save_path, conf['processed_data_path'])
    data_all = preprocess_data(raw_data_path, processed_data_path)
    split_by_domain(data_all, split_save_path, conf)


if __name__ == "__main__":
    # "categories": ["Digital Music", "Movies and TV", "Office Products", "Video Games", "Books"],
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="split config file", default="config_gender.json", required=False)
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    random.seed(config['seed'])
    split_to_domains(config)
