import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, BertConfig
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import numpy as np
import torch.nn.functional as F
from random import choice


class CLS_model(nn.Module):

    def __init__(self, pretrained_model_path, embedding_dim, target_size):
        super(CLS_model, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pretrained_model_path)
        # self.tokenizer  = BertTokenizer.from_pretrained('./prev_trained_model/roberta/')
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.fc1 = nn.Linear(768, embedding_dim)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(embedding_dim, target_size)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True)
        pooled_output = output[1]
        # layer = choice([7, 9, 12])
        hidden_states = output.hidden_states
        fc1_out = self.fc1(pooled_output)
        fc2_out = self.fc2(self.activation1(fc1_out))
        # hidden_states[6][:, 0]
        return fc2_out, hidden_states
        # return fc2_out, pooled_output


class MLP_BERT(nn.Module):

    def __init__(self, pretrained_model_path, embedding_dim, target_size):
        super(MLP_BERT, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pretrained_model_path)
        # self.tokenizer  = BertTokenizer.from_pretrained('./prev_trained_model/roberta/')
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.embed = self.bert.get_input_embeddings()
        self.fc1 = nn.Linear(768, embedding_dim)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(embedding_dim, target_size)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                aug_pooled=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True)
        embed = self.bert.get_input_embeddings()
        print(embed.shape)
        # return fc2_out, pooled_output


class DistilBERT(nn.Module):

    def __init__(self, pretrained_model_path, embedding_dim, target_size):
        super(DistilBERT, self).__init__()
        self.bert_config = DistilBertConfig.from_pretrained(
            pretrained_model_path)
        # self.tokenizer  = BertTokenizer.from_pretrained('./prev_trained_model/roberta/')
        self.bert = DistilBertModel.from_pretrained(pretrained_model_path)
        self.fc1 = nn.Linear(768, embedding_dim)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(embedding_dim, target_size)

    def forward(self,
                input_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True)
        pooled_output = output[0][:, 0, :]
        # print(pooled_output)
        # print(pooled_output.shape)
        # layer = choice([7, 9, 12])
        hidden_states = output.hidden_states
        fc1_out = self.fc1(pooled_output)
        fc2_out = self.fc2(self.activation1(fc1_out))
        # hidden_states[6][:, 0]
        return fc2_out, hidden_states


        # return fc2_out, pooled_output
class CNN_model(nn.Module):

    def __init__(self, args, target_size):
        super(CNN_model, self).__init__()
        self.args = args
        self.embedding_dim = args.cnn_embedding_dim
        self.target_size = target_size
        self.num_filters = args.num_filters
        self.filter_sizes = args.filter_sizes
        self.dropout = 0.5
        self.embedding = nn.Embedding(args.vocab_size, self.embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=self.num_filters,
                      kernel_size=fs) for fs in self.filter_sizes
        ])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(args.num_filters * len(args.filter_sizes),
                            self.target_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.fc(x)
        return output


class DAN_model(nn.Module):

    def __init__(self, args, target_size):
        super(DAN_model, self).__init__()
        self.args = args
        self.embedding_dim = args.cnn_embedding_dim
        self.target_size = target_size
        self.num_filters = args.num_filters
        self.filter_sizes = args.filter_sizes
        self.dropout = 0.5
        self.embedding = nn.Embedding(args.vocab_size, self.embedding_dim)

        for m in self.embedding.parameters():

            print(m.requires_grad)

        self.dropout1 = nn.Dropout(self.dropout)
        self.bn1 = nn.BatchNorm1d(600)
        self.fc1 = nn.Linear(600, 256)
        self.dropout2 = nn.Dropout(self.dropout)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, self.target_size)

    def forward(self, x):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x_max = x.max(dim=1)[0]
        x = x.mean(dim=1)

        x = torch.cat((x, x_max), dim=1)
        # print(x.shape)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        hidden_state = x
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x, hidden_state


class TinyBERT(nn.Module):

    def __init__(self, pretrained_model_path, embedding_dim, target_size):
        super(TinyBERT, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pretrained_model_path)
        # self.tokenizer  = BertTokenizer.from_pretrained('./prev_trained_model/roberta/')
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.fc1 = nn.Linear(312, embedding_dim)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(embedding_dim, target_size)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                aug_pooled=None):
        if aug_pooled is None:
            output = self.bert(input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
            pooled_output = output[1]
            fc1_out = self.fc1(pooled_output)
            fc2_out = self.fc2(self.activation1(fc1_out))
            return fc2_out, fc1_out


class DAN_model_test(nn.Module):

    def __init__(self, args, target_size):
        super(DAN_model_test, self).__init__()
        self.args = args
        self.embedding_dim = args.cnn_embedding_dim
        self.target_size = target_size
        self.num_filters = args.num_filters
        self.filter_sizes = args.filter_sizes
        self.dropout = 0.5
        self.embedding = nn.Embedding(args.vocab_size, self.embedding_dim)

        for m in self.embedding.parameters():

            print(m.requires_grad)

        self.dropout1 = nn.Dropout(self.dropout)
        self.bn1 = nn.BatchNorm1d(600)
        self.fc1 = nn.Linear(600, 256)
        self.dropout2 = nn.Dropout(self.dropout)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation2 = nn.ReLU()
        self.fc2 = nn.Linear(256, self.target_size)

    def forward(self, x):
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x_max = x.max(dim=1)[0]
        x = x.mean(dim=1)

        x = torch.cat((x, x_max), dim=1)
        hidden_state = x
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x, hidden_state


class Sprojector(nn.Module):

    def __init__(self, output_size=300):
        super(Sprojector, self).__init__()
        self.output_size = output_size

        self.linear1 = nn.Linear(600, self.output_size)
        self.linear2 = nn.Linear(600, self.output_size)
        self.linear3 = nn.Linear(600, self.output_size)
        self.linear4 = nn.Linear(600, self.output_size)
        self.activation = nn.Tanh()

    def forward(self, x):

        x1 = self.linear1(x)
        x1 = self.activation(x1)

        x2 = self.linear2(x)
        x2 = self.activation(x2)

        x3 = self.linear3(x)
        x3 = self.activation(x3)

        x4 = self.linear4(x)
        x4 = self.activation(x4)

        return x1, x2, x3, x4


class Tprojector(nn.Module):

    def __init__(self, output_size=300, layers=[5, 7, 9, 11]):
        super(Tprojector, self).__init__()
        self.output_size = output_size
        self.layer = layers
        self.linear1 = nn.Linear(768, self.output_size)
        self.linear2 = nn.Linear(768, self.output_size)
        self.linear3 = nn.Linear(768, self.output_size)
        self.linear4 = nn.Linear(768, self.output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):

        x1 = hidden_states[self.layer[0]][:, 0]
        x2 = hidden_states[self.layer[1]][:, 0]
        x3 = hidden_states[self.layer[2]][:, 0]
        x4 = hidden_states[self.layer[3]][:, 0]

        x1 = self.linear1(x1)
        x1 = self.activation(x1)

        x2 = self.linear2(x2)
        x2 = self.activation(x2)

        x3 = self.linear3(x3)
        x3 = self.activation(x3)

        x4 = self.linear4(x4)
        x4 = self.activation(x4)

        return x1, x2, x3, x4
