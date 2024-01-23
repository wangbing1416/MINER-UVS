import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args, shared_dim=128, sim_dim=64):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        # bert
        if self.args.dataset == 'weibo':
            bert_model = BertModel.from_pretrained('bert-base-chinese')
        else:
            bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.bert_hidden_size = args.bert_hidden_dim
        self.shared_text_linear = nn.Sequential(
            nn.Linear(self.bert_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        resnet = torchvision.models.resnet34(pretrained=True)

        num_ftrs = resnet.fc.out_features
        self.visualmodal = resnet
        self.shared_image = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

         # fusion
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )

        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))

        self.unimodal_classifier = nn.Sequential(
            nn.Linear(sim_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax()
        )

        self.fusion = nn.Sequential(
            nn.Linear(sim_dim * 2, sim_dim * 2),
            nn.BatchNorm1d(sim_dim * 2),
            nn.ReLU(),
            nn.Linear(sim_dim * 2, sim_dim),
            nn.ReLU()
        )

        self.sim_classifier = nn.Sequential(
            nn.Linear(sim_dim * 3, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2)
        )


    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image, mask):
        # IMAGE
        image = self.visualmodal(image)  # [N, 512]
        image_z = self.shared_image(image)
        image_z = self.image_aligner(image_z)
        image_pred = self.unimodal_classifier(image_z)

        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text_z = self.shared_text_linear(last_hidden_state)
        text_z = self.text_aligner(text_z)
        text_pred = self.unimodal_classifier(text_z)

        image_alpha = image_pred[:, -1] / (image_pred[:, -1] + text_pred[:, -1])  # normalize
        text_alpha = text_pred[:, -1] / (image_pred[:, -1] + text_pred[:, -1])

        text_image = torch.cat((text_alpha * text_z, image_alpha * image_z), 1)
        text_image = self.fusion(text_image)

        text_image_fusion = torch.cat((text_z, text_image, image_z), 1)  # TODO: order?

        # Fake or real
        class_output = self.sim_classifier(text_image_fusion)

        class_output = self.dropout(class_output)
        return class_output, image_pred, text_pred, text_image, image_z, text_z

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)