import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

LENGTH = 10
BATCH_SIZE = 100
NUM_BATCHES = 150
TOTAL_SIZE = BATCH_SIZE * NUM_BATCHES
TEST_SIZE = 2 ** LENGTH

LEARNING_RATE = 0.005
NUM_UNITS = 500
EPOCH = 100
LOG_EVERY_EPOCH = 5
LOG_EVERY_BATCH = NUM_BATCHES
NUM_CLASSES = 2
NUM_LAYERS = 5


def generate_all_data():
    data = torch.rand(TOTAL_SIZE, LENGTH).round()
    target = torch.fmod(torch.sum(data, 1), 2).squeeze()
    target = target.long()

    return data, target

def generate_data(size = BATCH_SIZE):
    data = torch.rand(size, LENGTH).round()
    target = torch.fmod(torch.sum(data, 1), 2).squeeze()
    target = target.long()

    return data, target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m0 = nn.Linear(LENGTH, NUM_UNITS)
        self.h0 = nn.Linear(LENGTH, NUM_UNITS)

        self.forget_lin = [nn.Linear(NUM_UNITS, NUM_UNITS).cuda() for i in range(NUM_LAYERS)]
        self.input_lin = [nn.Linear(NUM_UNITS, NUM_UNITS).cuda() for i in range(NUM_LAYERS)]
        self.candidate_lin = [nn.Linear(NUM_UNITS, NUM_UNITS).cuda() for i in range(NUM_LAYERS)]
        self.output_lin = [nn.Linear(NUM_UNITS, NUM_UNITS).cuda() for i in range(NUM_LAYERS)]

        self.activation = nn.Linear(NUM_UNITS + NUM_UNITS, 2)

    def step(self, h, m, forget_lin_, input_lin_, candidate_lin_, output_lin_):

        forget_gate = F.sigmoid(forget_lin_(h))
        input_gate = F.sigmoid(input_lin_(h))
        candidate = F.tanh(candidate_lin_(h))
        output_gate = F.sigmoid(output_lin_(h))

        m = (input_gate * candidate) + (forget_gate * m)
        h = F.tanh(output_gate * m)

        return h, m

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))

        m = self.m0(inputs)
        h = self.h0(inputs)

        for i in range(NUM_LAYERS):
            forget_lin_ = self.forget_lin[i]
            input_lin_ = self.input_lin[i]
            candidate_lin_ = self.candidate_lin[i]
            output_lin_ = self.output_lin[i]
            h, m = self.step(h, m, forget_lin_, input_lin_, candidate_lin_, output_lin_)

        outputs = F.log_softmax(self.activation(torch.cat((h, m), 1)))
        return outputs

model = Net()
model.cuda()

opt = optim.Adam(model.parameters())

d, t = generate_all_data()
idx = torch.randperm(TOTAL_SIZE)

def train(epoch):
    model.train()
    epoch_loss = 0
    idx = torch.randperm(TOTAL_SIZE)
    for i in range(NUM_BATCHES):
        batch_idx = idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        data, target = d[batch_idx], t[batch_idx]

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        opt.step()
        epoch_loss = loss.data[0]
    return epoch_loss

def test():
    model.eval()
    test_loss = 0
    correct = 0
    data, target = generate_data(size = TEST_SIZE)
    batches = int(TEST_SIZE / BATCH_SIZE)

    for i in range(batches):
        d = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        t = target[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        d, t = d.cuda(), t.cuda()
        d, t = Variable(d, volatile=True), Variable(t, volatile=True)
        out = model(d)
        test_loss += F.nll_loss(out, t).data[0]
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(t.data.view_as(pred)).cpu().sum()
    print(test_loss / batches)
    print(correct / TEST_SIZE)

for epoch in range(1, EPOCH + 1):
    loss = train(epoch)
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss))
test()
