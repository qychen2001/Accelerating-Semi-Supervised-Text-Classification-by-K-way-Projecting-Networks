import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset.cnn_dataset import build_vocab
from dataset.dataset import CLSDataset, UDADataset
from dataset.dataset import sup_collate_fn, unsup_collate_fn
from models.model import CLS_model, CNN_model, DAN_model_test, Tprojector, Sprojector
from utils.finetuning_argparse import get_argparse
from utils.get_tsa_thresh import get_tsa_thresh
from utils.utils import seed_everything, ProgressBar, init_logger, logger

from thop import profile
from thop import clever_format


def train_distill(args, sup_iter, unsup_iter, Smodel, Tmodel, Sprojector,
                  Tprojector, epoch):
    # 设置bert和cnn的训练器
    no_decay = ["bias", "LayerNorm.weight"]
    S_param = list(Smodel.named_parameters())
    S_grouped_param = [
        {
            'params':
            [p for n, p in S_param if not any(nd in n for nd in no_decay)],
            'weight_decay':
            args.weight_decay,
            'lr':
            args.cnn_learning_rate
        },
        {
            'params':
            [p for n, p in S_param if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.cnn_learning_rate
        },
    ]
    S_optimizer = torch.optim.AdamW(S_grouped_param,
                                    lr=args.cnn_learning_rate,
                                    eps=args.adam_epsilon)
    # 定义损失函数

    proj_param = list(Sprojector.parameters())
    proj_param.extend(list(Tprojector.parameters()))

    proj_optimizer = torch.optim.Adam(proj_param,
                                      lr=args.linear_learning_rate,
                                      weight_decay=args.weight_decay)

    S_batch_final_loss, T_batch_final_loss = 0, 0
    S_batch_sup_loss, T_batch_sup_loss = 0, 0
    S_batch_unsup_loss, T_batch_unsup_loss = 0, 0
    S_batch_distilling_loss, T_batch_distilling_loss = 0, 0
    Smodel.train()
    Tmodel.eval()
    pbar = ProgressBar(n_total=len(unsup_iter), desc='Training', width=10)
    n = len(unsup_iter)
    current_step = n * epoch
    logger.info("***** Running train %s *****")
    for step, unsup_batch in enumerate(unsup_iter):
        S_optimizer.zero_grad()

        for key in unsup_batch.keys():
            unsup_batch[key] = unsup_batch[key].to(args.device)
        try:
            sup_batch = next(sup_iter)
        except:
            sup_iter2 = iter(sup_iter)
            sup_batch = next(sup_iter2)
        for key in sup_batch.keys():
            sup_batch[key] = sup_batch[key].to(args.device)

        sup_criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)
        unsup_criterion = nn.KLDivLoss(reduction='none').to(args.device)

        Smodel.train()

        # print(sup_batch['all_cnn_input_ids'].shape)

        predictions, S_sup_hidden = Smodel(x=sup_batch['all_cnn_input_ids'])

        sup_loss = sup_criterion(predictions, sup_batch['all_labels'])
        if args.tsa:
            tsa_thresh = get_tsa_thresh(args.tsa,
                                        args.global_step,
                                        args.total_steps,
                                        start=1. / predictions.shape[-1],
                                        end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
            loss_mask = torch.ones_like(
                sup_batch['all_labels'], dtype=torch.float32) * (
                    1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / \
                torch.max(torch.sum(loss_mask, dim=-1),
                          torch.tensor(1.).to(args.device))
        else:
            sup_loss = torch.mean(sup_loss)
        # unsupervised loss
        uda_softmax_temp = args.uda_softmax_temp if args.uda_softmax_temp > 0 else 1.
        # ori
        len_ori = int(unsup_batch['all_input_ids'].shape[0] / 2)
        ori_logits, S_ori_hidden = Smodel(x=unsup_batch['cnn_ori_input_ids'])

        # confidence-based masking
        ori_prob = F.softmax(ori_logits, dim=-1)
        if args.uda_confidence_thresh != -1:
            unsup_loss_mask = torch.max(ori_prob,
                                        dim=-1)[0] > args.uda_confidence_thresh
            unsup_loss_mask = unsup_loss_mask.type(torch.float32)
        else:
            unsup_loss_mask = torch.ones(len_ori, dtype=torch.float32)
        unsup_loss_mask = unsup_loss_mask.to(args.device)
        # Sharpening Predictions
        ori_prob = F.softmax(ori_logits / uda_softmax_temp, dim=-1)
        # aug
        ori_prob = ori_prob.detach()  # 将prob从计算图中脱离，不计算梯度
        logits, S_aug_hidden = Smodel(x=unsup_batch['cnn_aug_input_ids'])
        aug_log_prob = F.log_softmax(logits, dim=-1)
        # KLdiv loss
        unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask,
                               dim=-1) / torch.max(
                                   torch.sum(unsup_loss_mask, dim=-1),
                                   torch.tensor(1.).to(args.device))

        uda_loss = sup_loss + unsup_loss

        with torch.no_grad():
            T_sup_logits, T_sup_hidden = Tmodel(
                input_ids=sup_batch['all_input_ids'],
                attention_mask=sup_batch['all_attention_mask'],
                token_type_ids=sup_batch['all_token_type_ids'])
            T_aug_logits, T_aug_hidden = Tmodel(
                input_ids=unsup_batch['all_input_ids'][len_ori:],
                attention_mask=unsup_batch['all_attention_mask'][len_ori:],
                token_type_ids=unsup_batch['all_token_type_ids'][len_ori:])

        S_sup_loss = F.mse_loss(predictions, T_sup_logits, reduction='mean')
        S_unsup_loss = F.mse_loss(logits, T_aug_logits, reduction='mean')

        output_distill_loss = S_sup_loss + S_unsup_loss

        T_sup_proj = Tprojector(T_sup_hidden)
        T_unsup_proj = Tprojector(T_aug_hidden)

        S_sup_proj = Sprojector(S_sup_hidden)
        S_unsup_proj = Sprojector(S_aug_hidden)

        hidden_distill_loss = 0

        # check the size

        # assert (T_sup_proj.size() == S_sup_proj.size())
        # assert (T_sup_proj.size() == T_unsup_proj.size())
        # assert (S_sup_proj.size() == S_unsup_proj.size())
        for i in range(len(T_sup_proj)):
            hidden_distill_loss += F.mse_loss(T_sup_proj[i], S_sup_proj[i])
            hidden_distill_loss += F.mse_loss(T_unsup_proj[i], S_unsup_proj[i])

        # 打印信息
        S_final_loss = output_distill_loss

        S_final_loss.backward(retain_graph=True)
        hidden_distill_loss.backward()
        S_optimizer.step()
        proj_optimizer.step()

        # 梯度下降，更新参数
        S_batch_final_loss += S_final_loss.item()
        S_batch_sup_loss += S_sup_loss.item()
        S_batch_unsup_loss += S_unsup_loss.item()
        pbar(
            step, {
                'global_S_final': S_batch_final_loss / (args.global_step + 1),
                'global_S_sup': S_batch_sup_loss / (args.global_step + 1),
                'global_S_unsup': S_batch_unsup_loss / (args.global_step + 1)
            })
        args.global_step += 1


def evaluate(args, eval_iter, model, type='bert'):
    # DONE:将CNN和BERT的评估写在一起
    logger.info("***** Running Evalation *****")
    eval_loss = 0.0
    eval_steps = 0
    criterion = nn.CrossEntropyLoss().to(args.device)
    pbar = ProgressBar(n_total=len(eval_iter), desc="Evaluating")
    pres, trues = [], []
    model.eval()
    for step, batch in enumerate(eval_iter):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device)
        with torch.no_grad():
            if type != 'bert':
                predictions, _ = model(x=batch['all_cnn_input_ids'])
            else:
                predictions, _ = model(
                    input_ids=batch['all_input_ids'],
                    attention_mask=batch['all_attention_mask'],
                    token_type_ids=batch['all_token_type_ids'])

        loss = criterion(predictions, batch['all_labels'])
        eval_loss += loss.item()
        eval_steps += 1

        pbar(step)
        _, pre = torch.max(predictions, axis=1)
        pre = pre.cpu().numpy().tolist()
        true = batch['all_labels'].cpu().numpy().tolist()
        pres.extend(pre)
        trues.extend(true)

    score_f1 = f1_score(trues, pres, average="macro")
    accuracy = accuracy_score(trues, pres)
    eval_loss = eval_loss / eval_steps
    logger.info("Macro F1:{:.4f},Accuracy:{:.4f},Loss:{:.4f}".format(
        score_f1, accuracy, eval_loss))
    return score_f1, accuracy


def main():
    args = get_argparse().parse_args()
    args.datapath = os.path.join("./data", args.task)
    print(
        json.dumps(vars(args),
                   sort_keys=True,
                   indent=4,
                   separators=(', ', ': '),
                   ensure_ascii=False))

    seed_everything(args.seed)

    log_file = 'Proj {}.log'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    log_path = './log/{}/{}/{}/'.format(args.task, args.num_sup, 'Proj_DAN')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, log_file)
    init_logger(log_file)

    # 设置保存目录
    if not os.path.exists(
            os.path.join(args.output_dir, args.task, str(args.num_sup),
                         'Proj_DAN')):
        os.makedirs(
            os.path.join(args.output_dir, args.task, str(args.num_sup),
                         'Proj_DAN'))
    args.output_dir = os.path.join(args.output_dir, args.task,
                                   str(args.num_sup), 'Proj_DAN')
    # device
    args.device = torch.device(
        "cuda:{}".format(args.gpu_num) if torch.cuda.is_available() else "cpu")
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer
    vocab = build_vocab(args)  # 构建词典
    sup_dataset = CLSDataset(args,
                             csv_path="./data/{}/train_{}.csv".format(
                                 args.task, str(args.num_sup)),
                             tokenizer=args.tokenizer,
                             type="train",
                             vocab=vocab)
    num_labels = sup_dataset.df["label"].nunique()
    args.num_labels = num_labels
    unsup_dataset = UDADataset(args,
                               csv_path="./data/{}/unsup.csv".format(
                                   args.task),
                               tokenizer=args.tokenizer,
                               vocab=vocab)
    eval_dataset = CLSDataset(args,
                              csv_path="./data/{}/dev.csv".format(args.task),
                              tokenizer=args.tokenizer,
                              type="eval",
                              vocab=vocab)
    test_dataset = CLSDataset(args,
                              csv_path="./data/{}/test.csv".format(args.task),
                              tokenizer=args.tokenizer,
                              type="test",
                              vocab=vocab)
    sup_iter = DataLoader(sup_dataset,
                          shuffle=True,
                          batch_size=args.per_gpu_train_batch_size,
                          collate_fn=sup_collate_fn)
    unsup_iter = DataLoader(unsup_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=unsup_collate_fn)
    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=sup_collate_fn)
    test_iter = DataLoader(test_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=sup_collate_fn)
    Smodel = DAN_model_test(args, num_labels)
    Smodel.embedding.weight.data = vocab.vectors

    # stat(Smodel(1,256))

    Tmodel = CLS_model(args.model_name_or_path, 256, num_labels)

    Smodel.to(args.device)
    Tmodel.to(args.device)
    Tmodel.load_state_dict(
        torch.load("output/{}/{}/UDA_BERT/best_model.pkl".format(
            args.task, args.num_sup)))
    print("load teacher model success")

    # input = torch.ones(1, 256).long()
    # flops, params = profile(Smodel.cpu(), inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)

    Tproj = Tprojector(output_size=args.projector_hidden_size)
    Sproj = Sprojector(output_size=args.projector_hidden_size)

    Tproj.to(args.device)
    Sproj.to(args.device)

    args.global_step = 0
    S_best_acc = 0
    early_stop = 0
    logger.info(args)
    # _, t_acc = evaluate(args, eval_iter, Tmodel, type='bert')
    # print(t_acc)
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        Smodel.train()
        Tmodel.eval()
        # add "epoch" to training function
        train_distill(args, sup_iter, unsup_iter, Smodel, Tmodel, Sproj, Tproj,
                      epoch)
        _, S_eval_acc = evaluate(args, eval_iter, Smodel, type='cnn')
        if S_eval_acc > S_best_acc:
            early_stop = 0
            S_best_acc = S_eval_acc
            logger.info(
                "the best student eval acc is {:.4f}, saving model !!".format(
                    S_best_acc))
            S_best_model = copy.deepcopy(
                Smodel.module if hasattr(Smodel, "module") else Smodel)
            torch.save(S_best_model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pkl"))
        else:
            early_stop += 1
        if early_stop >= args.early_stop:
            logger.info("Early stop in {} epoch!".format(epoch))
            break
    _, S_acc = evaluate(args, test_iter, S_best_model, type='cnn')
    logger.info("Student Test acc is {:.4f}".format(S_acc))


if __name__ == "__main__":
    main()
