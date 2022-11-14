from distutils.command.config import config
import imp
import random

import torch
from augment import eda
from torch.utils.data import Dataset
import pandas as pd
from dataset.cnn_dataset import convert_to_cnn, convert_to_unsup_cnn


class CLSDataset(Dataset):
    def __init__(self, args, vocab, csv_path, tokenizer, type="train"):
        self.df = pd.read_csv(csv_path)
        self.max_seq_length = args.train_max_seq_length
        self.tokenizer = tokenizer
        self.type = type
        self.config = args
        self.vocab = vocab

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):
        content = self.df.loc[index, "sentence"]
        label = self.df.loc[index, "label"]
        # d_encode = self.tokenizer.encode_plus(content)
        # padding
        sample_cnn = convert_to_cnn(
            content=content, vocab=self.vocab, args=self.config)  # CNN
        d_encode = self.tokenizer.encode_plus(content,
                                              padding="max_length",
                                              max_length=self.max_seq_length,
                                              truncation=True)
        sample = {"input_ids": d_encode['input_ids'],
                  "attention_mask": d_encode['attention_mask'],
                  "length": sum(d_encode['attention_mask']),
                  "label": label}
        sample.update(sample_cnn)
        return sample


class UDADataset(Dataset):
    def __init__(self, args, csv_path, tokenizer, vocab):
        self.df = pd.read_csv(csv_path)
        self.max_seq_length = args.train_max_seq_length
        #self.smw = Similarword(create_num=2, change_rate=0.3)
        self.tokenizer = tokenizer
        self.config = args
        self.vocab = vocab
        #self.aug = eda.eda()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):
        content = self.df.loc[index, "sentence"]
        aug = self.df.loc[index, "aug"]
        contents = []
        # TODO: use the augmented data instead of EDA
        contents.append(content)
        contents.append(aug)
        # try:
        #     contents.append(eda.eda(content)[1])
        # # contents保存增强后的文本，最多两条，如果只有一条则把原来的再保存一遍
        # except:
        #     contents.append(contents[0])
        sample = {}
        # 更新cnn的数据集
        sample.update(convert_to_unsup_cnn(
            contents[0], contents[1], self.vocab, self.config))
        # contents内包含两条文本，其中index=0的为原始文本，index=1的为增强后的文本
        for index, content in enumerate(contents):
            d_encode = self.tokenizer.encode_plus(content)
            # padding
            if self.max_seq_length >= len(d_encode['input_ids']):
                length = len(d_encode['input_ids'])
                padding_length = self.max_seq_length - \
                    len(d_encode['input_ids'])
                d_encode['input_ids'] += [self.tokenizer.pad_token_id] * \
                    padding_length
                d_encode['attention_mask'] += [0] * padding_length
            elif self.max_seq_length < len(d_encode['input_ids']):
                length = self.max_seq_length
                d_encode['input_ids'] = d_encode['input_ids'][:self.max_seq_length]
                d_encode['attention_mask'] = d_encode['attention_mask'][:self.max_seq_length]
            # 断言，判断是否异常
            assert len(d_encode['input_ids']) == self.max_seq_length
            assert len(d_encode['attention_mask']) == self.max_seq_length

            if index == 0:
                sample.update({
                    "ori_input_ids": d_encode['input_ids'],
                    "ori_attention_mask": d_encode['attention_mask'],
                    "ori_length": length,
                })
            elif index == 1:
                sample.update({
                    "aug_input_ids": d_encode['input_ids'],
                    "aug_attention_mask": d_encode['attention_mask'],
                    "aug_length": length,
                })
        return sample


def sup_collate_fn(batch):
    max_len = max([x['length'] for x in batch])
    input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    attention_mask = torch.tensor(
        [x['attention_mask'][:max_len] for x in batch])
    label = torch.tensor([x["label"] for x in batch])

    cnn_input_ids = torch.tensor(
        [x['cnn_input_ids'][:max_len] for x in batch])
    return {"all_input_ids": input_ids,
            "all_attention_mask": attention_mask,
            "all_labels": label,
            "all_cnn_input_ids": cnn_input_ids
            }


def unsup_collate_fn(batch):
    ori_length = [x['ori_length'] for x in batch]
    aug_length = [x['aug_length'] for x in batch]
    ori_length.extend(aug_length)
    max_len = max(ori_length)
    # bert输入
    ori_input_ids = torch.tensor(
        [x['ori_input_ids'][:max_len] for x in batch], dtype=torch.long)
    aug_input_ids = torch.tensor(
        [x['aug_input_ids'][:max_len] for x in batch], dtype=torch.long)

    ori_attention_mask = torch.tensor(
        [x['ori_attention_mask'][:max_len] for x in batch], dtype=torch.long)
    aug_attention_mask = torch.tensor(
        [x['aug_attention_mask'][:max_len] for x in batch], dtype=torch.long)
    # ori
    cnn_ori_input_ids = torch.tensor(
        [x['cnn_ori_input_ids'][:max_len] for x in batch], dtype=torch.long)
    # aug
    cnn_aug_input_ids = torch.tensor(
        [x['cnn_aug_input_ids'][:max_len] for x in batch], dtype=torch.long)

    all_input_ids = torch.cat([ori_input_ids, aug_input_ids], dim=0)
    all_attention_mask = torch.cat(
        [ori_attention_mask, aug_attention_mask], dim=0)

    return {"all_input_ids": all_input_ids,
            "all_attention_mask": all_attention_mask,
            "cnn_ori_input_ids": cnn_ori_input_ids,
            "cnn_aug_input_ids": cnn_aug_input_ids}
