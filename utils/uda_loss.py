import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.get_tsa_thresh import get_tsa_thresh


def get_bert_uda_loss(args, sup_batch, unsup_batch, model):
    sup_criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)
    unsup_criterion = nn.KLDivLoss(reduction='none').to(args.device)

    model.train()
    predictions, _ = model(
        input_ids=sup_batch['all_input_ids'],
        attention_mask=sup_batch['all_attention_mask'],
        token_type_ids=sup_batch['all_token_type_ids'])
    sup_loss = sup_criterion(predictions, sup_batch['all_labels'])
    if args.tsa:
        tsa_thresh = get_tsa_thresh(args.tsa,
                                    args.global_step,
                                    args.total_steps,
                                    start=1. / predictions.shape[-1],
                                    end=1)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
        loss_mask = torch.ones_like(sup_batch['all_labels'], dtype=torch.float32) * (
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
    ori_logits, _ = model(
        input_ids=unsup_batch['all_input_ids'][:len_ori],
        attention_mask=unsup_batch['all_attention_mask'][:len_ori],
        token_type_ids=unsup_batch['all_token_type_ids'][:len_ori]
    )

    # confidence-based masking
    ori_prob = F.softmax(ori_logits, dim=-1)
    if args.uda_confidence_thresh != -1:
        unsup_loss_mask = torch.max(
            ori_prob, dim=-1)[0] > args.uda_confidence_thresh
        unsup_loss_mask = unsup_loss_mask.type(torch.float32)
    else:
        unsup_loss_mask = torch.ones(len_ori, dtype=torch.float32)
    unsup_loss_mask = unsup_loss_mask.to(args.device)
    # Sharpening Predictions
    ori_prob = F.softmax(ori_logits / uda_softmax_temp, dim=-1)
    # aug
    ori_prob = ori_prob.detach()  # 将prob从计算图中脱离，不计算梯度
    logits, _ = model(
        input_ids=unsup_batch['all_input_ids'][len_ori:],
        attention_mask=unsup_batch['all_attention_mask'][len_ori:],
        token_type_ids=unsup_batch['all_token_type_ids'][len_ori:]
    )
    aug_log_prob = F.log_softmax(logits, dim=-1)
    # KLdiv loss
    unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
    unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                             torch.tensor(1.).to(args.device))

    final_loss = sup_loss + args.uda_coeff * unsup_loss

    return predictions, ori_logits, final_loss


def get_cnn_uda_loss(args, sup_batch, unsup_batch, model):
    sup_criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)
    unsup_criterion = nn.KLDivLoss(reduction='none').to(args.device)

    model.train()
    predictions = model(x=sup_batch['all_cnn_input_ids'])
    sup_loss = sup_criterion(predictions, sup_batch['all_labels'])
    if args.tsa:
        tsa_thresh = get_tsa_thresh(args.tsa,
                                    args.global_step,
                                    args.total_steps,
                                    start=1. / predictions.shape[-1],
                                    end=1)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
        loss_mask = torch.ones_like(sup_batch['all_labels'], dtype=torch.float32) * (
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
    ori_logits = model(x=unsup_batch['cnn_ori_input_ids'])

    # confidence-based masking
    ori_prob = F.softmax(ori_logits, dim=-1)
    if args.uda_confidence_thresh != -1:
        unsup_loss_mask = torch.max(
            ori_prob, dim=-1)[0] > args.uda_confidence_thresh
        unsup_loss_mask = unsup_loss_mask.type(torch.float32)
    else:
        unsup_loss_mask = torch.ones(len_ori, dtype=torch.float32)
    unsup_loss_mask = unsup_loss_mask.to(args.device)
    # Sharpening Predictions
    ori_prob = F.softmax(ori_logits / uda_softmax_temp, dim=-1)
    # aug
    ori_prob = ori_prob.detach()  # 将prob从计算图中脱离，不计算梯度
    logits = model(x=unsup_batch['cnn_aug_input_ids'])
    aug_log_prob = F.log_softmax(logits, dim=-1)
    # KLdiv loss
    unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
    unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                             torch.tensor(1.).to(args.device))

    final_loss = sup_loss + args.uda_coeff * unsup_loss

    return predictions, ori_logits, final_loss


def get_optimizer(args, Tmodel, Smodel):
    no_decay = ["bias", "LayerNorm.weight"]
    S_param = list(Smodel.named_parameters())
    S_grouped_param = [
        {'params': [p for n, p in S_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.cnn_learning_rate},
        {'params': [p for n, p in S_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.cnn_learning_rate},
    ]
    S_optimizer = torch.optim.AdamW(S_grouped_param,
                                    lr=args.cnn_learning_rate,
                                    eps=args.adam_epsilon)
    T_bert_param = list(Tmodel.bert.named_parameters())
    T_linear_param = list(Tmodel.fc1.named_parameters())
    T_linear_param.extend(list(Tmodel.fc2.named_parameters()))
    T_grouped_param = [
        {'params': [p for n, p in T_bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in T_bert_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in T_linear_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.linear_learning_rate},
        {'params': [p for n, p in T_linear_param if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.linear_learning_rate},
    ]
    T_optimizer = torch.optim.AdamW(T_grouped_param,
                                    lr=args.learning_rate,
                                    eps=args.adam_epsilon)

    T_scheduler = torch.optim.lr_scheduler.MultiStepLR(T_optimizer, milestones=[args.bert_lr_1, args.bert_lr_2],
                                                       gamma=0.1)  # 3，10
    S_scheduler = torch.optim.lr_scheduler.MultiStepLR(S_optimizer, milestones=[args.cnn_lr_1, args.cnn_lr_2],
                                                       gamma=0.1)

    return T_optimizer, S_optimizer, T_scheduler, S_scheduler
