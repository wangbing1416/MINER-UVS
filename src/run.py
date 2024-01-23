import numpy as np
import argparse
import logging
import os, sys
from time import strftime, localtime
import tqdm
import random

# import process_weibo as process_data
import process_gossipcop as process_data

import copy
import pickle as pickle
from random import sample
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from transformers import BertModel, BertTokenizer
from model import CNN_Fusion
from loss import ContrastiveLoss

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, Image: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is " + str(len(train[i])))
        print(i)
        # print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is " + str(len(train[3])))

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is " + str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


def main(args):
    print('loading data')
    train, validation, test = load_data(args)
    test_id = test['post_id']

    train_dataset = Rumor_Data(train)
    validate_dataset = Rumor_Data(validation)
    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.number_workers, shuffle=False)

    logger.info('building model')
    model = CNN_Fusion(args)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    logger.info("loader size " + str(len(train_loader)))
    best_validate_f1 = 0.000
    early_stop = 0
    best_validate_dir = ''

    logger.info('begin training...')
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    contrastiveLoss = ContrastiveLoss(batch_size=args.batch_size, temperature=args.temp)

    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ["bertModel.embeddings", "bertModel.encoder"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=args.adam_epsilon)

    for epoch in range(args.num_epochs):
        p = float(epoch) / 100
        lr = args.learning_rate / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        cost_vector = []
        acc_vector = []

        for i, (train_data, train_labels, event_labels) in tqdm.tqdm(enumerate(train_loader)):
            train_text, train_image, train_mask, train_labels, event_labels = to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), to_var(train_labels), to_var(event_labels)
            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, image_pred, text_pred, _, image_z, text_z = model(train_text, train_image, train_mask)
            loss_cl = contrastiveLoss(image_z, text_z)

            train_labels_unlabeled = to_var(torch.ones(train_labels.shape[0])) - train_labels
            image_pred_phi = image_pred[:, 0]
            text_pred_phi = text_pred[:, 0]
            # 0 - real, 1 - fake
            loss_pu_image = torch.log(torch.sum(train_labels * (image_pred_phi + 1e-6))) - torch.log(torch.sum(train_labels)) - \
                            torch.sum(train_labels_unlabeled * torch.log(image_pred_phi + 1e-6)) / torch.sum(train_labels_unlabeled)
            loss_pu_text = torch.log(torch.sum(train_labels * (text_pred_phi + 1e-6))) - torch.log(torch.sum(train_labels)) - \
                           torch.sum(train_labels_unlabeled * torch.log(text_pred_phi + 1e-6)) / torch.sum(train_labels_unlabeled)
            loss_pu = loss_pu_image + loss_pu_text
            if not (100 > loss_pu > -100):
                logger.info(
                    'loss_pu_image: {}, \nloss_pu_text: {}, \n image_pred_phi: {}, \n text_pred_phi: {}, \n train_labels: {}, \n train_labels_unlabeled：{}'.format(
                        loss_pu_image, loss_pu_text, image_pred_phi, text_pred_phi, train_labels, train_labels_unlabeled))
                continue
            loss_uvc = loss_pu + args.gamma * loss_cl

            # Fake or Real loss
            loss_mvc = criterion(class_outputs, train_labels)
            loss = loss_mvc + args.balanced * loss_uvc

            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            accuracy = (train_labels == argmax.squeeze()).float().mean()
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        model.eval()
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), to_var(validate_labels), to_var(event_labels)
            validate_outputs, _, _, _, _, _ = model(validate_text, validate_image, validate_mask)

            _, validate_argmax = torch.max(validate_outputs, 1)
            if i == 0:
                validate_score = to_np(validate_outputs.squeeze())
                validate_pred = to_np(validate_argmax.squeeze())
                validate_true = to_np(validate_labels.squeeze())
            else:
                validate_score = np.concatenate((validate_score, to_np(validate_outputs.squeeze())), axis=0)
                validate_pred = np.concatenate((validate_pred, to_np(validate_argmax.squeeze())), axis=0)
                validate_true = np.concatenate((validate_true, to_np(validate_labels.squeeze())), axis=0)

        validate_accuracy = metrics.accuracy_score(validate_true, validate_pred)
        validate_f1 = metrics.f1_score(validate_true, validate_pred, average='macro')
        logger.info('Epoch [%d/%d], Loss: %.4f, Train_Acc: %.4f, Validate_Acc: %.4f, Validate_F1: : %.4f.' %
                    (epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(acc_vector), validate_accuracy, validate_f1))
        if epoch > 10:
            best_validate_dir = args.output_file + args.id + '.pkl'
            if validate_f1 > best_validate_f1:
                early_stop = 0
                best_validate_f1 = validate_f1
                if not os.path.exists(args.output_file):
                    os.mkdir(args.output_file)
                torch.save(model.state_dict(), best_validate_dir)
            else:
                early_stop += 1
                if early_stop == args.early_stop_epoch:
                    break


    # Test the Model
    logger.info('testing model')
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels)
        test_outputs, image_outputs, text_outputs, _, _, _ = model(test_text, test_image, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    logger.info("Classification Acc: %.4f, AUC-ROC: %.4f" % (test_accuracy, test_aucroc))
    logger.info("Classification report:\n%s\n" % (metrics.classification_report(test_true, test_pred, digits=4)))
    logger.info("Classification confusion matrix:\n%s\n" % (test_confusion_matrix))


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
    for i, l in enumerate(label):
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def re_tokenize_sentence(flag, max_length, dataset):
    if dataset == 'weibo':
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)[:max_length]
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask


def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)
    re_tokenize_sentence(train, max_length=args.max_length, dataset=args.dataset)
    re_tokenize_sentence(validate, max_length=args.max_length, dataset=args.dataset)
    re_tokenize_sentence(test, max_length=args.max_length, dataset=args.dataset)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gossip', help='')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--number_workers', type=int, default=4, help='')

    parser.add_argument('--max_length', type=int, default=200, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--early_stop_epoch', type=int, default=10, help='')

    parser.add_argument('--temp', type=float, default=0.2, help='')
    parser.add_argument('--gamma', type=float, default=0.0, help='corf of pretraining loss')
    parser.add_argument('--balanced', type=float, default=0.01, help='corf of pretraining loss')

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--bert_lr', type=float, default=0.00003, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    args = parser.parse_args()
    args.output_file = '../Data/' + args.dataset + '/RESULT_text_image/'
    args.id = '{}-{}.log'.format(args.dataset, strftime("%y%m%d-%H%M", localtime()))
    log_file = '../log/' + args.id
    logger.addHandler(logging.FileHandler(log_file))

    # write arguments into the logger
    logger.info('> training arguments:')
    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    main(args)


