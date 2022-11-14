import pickle as pkl
import numpy as np
import os
from tqdm import tqdm
import argparse
import pandas as pd
from torchtext.legacy import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm

# 使用torchtext构建cnn的输入


def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def tokenize(x): return x.split()


def get_dataset(csv_data, text_field, label_field, unsup=False):

    fields = [("sentence", text_field), ("label", label_field)]
    examples = []
    if unsup:
        for text in tqdm(csv_data['sentence']):
            examples.append(data.Example.fromlist([text, None], fields))
    else:
        for text, label in tqdm(zip(csv_data['sentence'], csv_data['label'])):
            examples.append(data.Example.fromlist([text, label], fields))
    return examples, fields


def build_vocab(args, glove_path=None):
    LABEL = data.Field(sequential=False, use_vocab=False)
    TEXT = data.Field(sequential=True, tokenize=tokenize,
                      lower=True)

    csv_path_train = "./data/{}/train_{}.csv".format(
        args.task, str(args.num_sup))
    csv_path_unsup = "./data/{}/unsup.csv".format(args.task)
    train_data, unsup_data = pd.read_csv(
        csv_path_train), pd.read_csv(csv_path_unsup)

    train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
    unsup_examples, unsup_fields = get_dataset(
        unsup_data, TEXT, None, unsup=True)
    # 构建Dataset数据集
    train = data.Dataset(train_examples, train_fields)
    unsup = data.Dataset(unsup_examples, unsup_fields)
    if glove_path is not None:
        cache = glove_path
        if not os.path.exists(cache):
            os.mkdir(cache)
        args.glove_path = glove_path
    else:
        cache = args.glove_path  # 如果glove_path为None，则使用默认的glove_path
    vectors = Vectors(name='glove.6B.300d.txt', cache=cache)
    # 指定 Vector 缺失值的初始化方式，没有命中的token的初始化方式
    TEXT.build_vocab(train, unsup, vectors=vectors)  # 使用训练集和无监督集来构建词典
    # 将参数中的vocab大小
    args.vocab_size = len(TEXT.vocab)
    return TEXT.vocab


def convert_to_dan(content, vocab, args):
    pad_size = args.train_max_seq_length
    tokens = tokenize(content)
    # 将句子处理为等长
    if pad_size:
        if len(tokens) < pad_size:
            tokens.extend(['<pad>'] * (pad_size - len(tokens)))
        else:
            tokens = tokens[:pad_size]
    words_line = []
    for word in tokens:
        words_line.append(vocab.stoi[word])

    buckets = 250499
    bigram = []
    trigram = []
    # ------ngram------
    for i in range(pad_size):
        bigram.append(biGramHash(words_line, i, buckets))
        trigram.append(triGramHash(words_line, i, buckets))

    sample = {"dan_input_ids": words_line,
              "bigram": bigram,
              "trigram": trigram}
    return sample


def convert_to_unsup_dan(ori, aug, vocab, args):
    pad_size = args.train_max_seq_length
    token_ori = tokenize(ori)
    token_aug = tokenize(aug)
    if pad_size:
        if len(token_ori) < pad_size:
            token_ori.extend(['<pad>'] * (pad_size - len(token_ori)))
        else:
            token_ori = token_ori[:pad_size]

        if len(token_aug) < pad_size:
            token_aug.extend(['<pad>'] * (pad_size - len(token_aug)))
        else:
            token_aug = token_aug[:pad_size]

    words_line_ori = []
    words_line_aug = []
    for word in token_ori:
        words_line_ori.append(vocab.stoi[word])

    for word in token_aug:
        words_line_aug.append(vocab.stoi[word])

    bigram_ori = []
    trigram_ori = []
    bigram_aug = []
    trigram_aug = []
    buckets = 250499
    for i in range(pad_size):
        bigram_ori.append(biGramHash(words_line_ori, i, buckets))
        trigram_ori.append(triGramHash(words_line_ori, i, buckets))
        bigram_aug.append(biGramHash(words_line_aug, i, buckets))
        trigram_aug.append(triGramHash(words_line_aug, i, buckets))

    sample = {"ori_dan_input_ids": words_line_ori,
              "aug_dan_input_ids": words_line_aug,
              "ori_bigram": bigram_ori,
              "ori_trigram": trigram_ori,
              "aug_bigram": bigram_aug,
              "aug_trigram": trigram_aug}
    return sample


# DONE:已完成将数据转化为CNN的输入

def build_vocab_sup(args, glove_path=None):
    LABEL = data.Field(sequential=False, use_vocab=False)
    TEXT = data.Field(sequential=True, tokenize=tokenize,
                      lower=True)  # 构建filed

    # 得到任务相关的路径，并且读入csv文件
    csv_path_train = "./data/{}/train_sup.csv".format(
        args.task, str(args.num_sup))
    train_data = pd.read_csv(csv_path_train)

    # 得到构建Dataset所需的examples和fields
    train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
    # 构建Dataset数据集
    train = data.Dataset(train_examples, train_fields)
    if glove_path is not None:
        cache = glove_path
        if not os.path.exists(cache):
            os.mkdir(cache)
        args.glove_path = glove_path
    else:
        cache = args.glove_path  # 如果glove_path为None，则使用默认的glove_path
    vectors = Vectors(name='glove.6B.300d.txt', cache=cache)
    # 指定 Vector 缺失值的初始化方式，没有命中的token的初始化方式
    TEXT.build_vocab(train, vectors=vectors)  # 使用训练集和无监督集来构建词典
    # 将参数中的vocab大小
    args.vocab_size = len(TEXT.vocab)
    return TEXT.vocab
