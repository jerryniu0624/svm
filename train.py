import torch
from sklearn import svm
from dataset import OmniglotPair, Omniglot, get_loaders
from model import SiameseSVMNet
from svmloss import SVMLoss, compute_accuracy
import random
from torch.autograd import Variable
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import draw as d
import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse

# constants
dim = 105

loss1 = []
acc1 = []

def draw_loss(loss1,name):
    x=range(0,len(loss1))
    plt.title("loss of the Svm") 
    plt.xlabel("training times") 
    plt.ylabel("loss") 
    plt.plot(x,loss1)
    plt.legend() # 添加图例
    plt.savefig('./'+name+'.jpg')
    plt.show()
    plt.close()


def draw_acc(acc,name):
    x=range(0,len(acc))
    plt.title("accuracy of the Svm")
    plt.xlabel("training times")
    plt.ylabel("accuracy")
    plt.plot(x,acc)
    plt.legend() # 添加图例
    plt.savefig('./'+name+'.jpg')
    plt.show()
    plt.close()

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese kernel SVM')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--C', type=float, default=0.2, metavar='C',
                        help='C regulation coefficients (default: 0.2)')
    parser.add_argument('--test-number', type=int, default=10, metavar='N',
                        help='number of different subset of test (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    args = get_args()
    model = SiameseSVMNet()
    if args.cuda:
        model = model.cuda()
    criterion = SVMLoss(args.C)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, validate_loader, test_data = get_loaders(args)

    def training(epoch):
        print('Epoch', epoch + 1)
        model.train()
        for batch_idx, (x0, x1, label) in enumerate(train_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            optimizer.zero_grad()
            output = model(x0, x1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                loss1.append(loss.item())
                print("\n Batch:  ", batch_idx, " / ",
                      len(train_loader), " --- Loss: ", loss.item())

    def validate():
        model.eval()
        acc = 0
        for batch_idx, (x0, x1, label) in enumerate(validate_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            output = model(x0, x1)
            acc += compute_accuracy(output, label)

        acc = 100.0 * acc / len(validate_loader.dataset)
        print('\nValidation set: Accuracy: {}%\n'.format(acc))
        acc1.append(acc)
        return acc


    # def test(n, k):
    #     model.eval()
    #     clf = svm.SVC(C=args.C, kernel='linear')
    #     featuremodel = model.get_FeatureNet()
    #     if args.cuda:
    #         featuremodel = featuremodel.cuda()
    #     # choose classes
    #     acc = 0
    #     for i in range(args.test_number):
    #         random.seed(i)
    #         temp_ = []
    #         for i in range(200 - 0):
    #             temp_.append(i)
    #         random.shuffle(temp_)
    #         choosen_classes = temp_[:n]

    #         X_train = []
    #         y_train = []
    #         y_test = []
    #         X_test = []
    #         for cl in choosen_classes:
    #             for i in range(k):
    #                 X_train.append(test_data[cl * 20 + i][0])
    #                 if args.cuda:
    #                     X_train[-1] = X_train[-1].cuda()
    #                 y_train.append(cl)
    #             for i in range(k, 20):
    #                 X_test.append(test_data[cl * 20 + i][0])
    #                 if args.cuda:
    #                     X_test[-1] = X_test[-1].cuda()
    #                 y_test.append(cl)

    #         # calculate features
    #         train_features = []
    #         test_features = []
    #         for train_point in X_train:
    #             train_features.append(featuremodel(
    #                 Variable(train_point)).cpu().data.numpy())
    #         for test_point in X_test:
    #             test_features.append(featuremodel(
    #                 Variable(test_point)).cpu().data.numpy())

    #         # create features
    #         train_features = np.array(train_features)
    #         train_features = np.reshape(
    #             train_features, (train_features.shape[0], 4096))
    #         test_features = np.array(test_features)
    #         test_features = np.reshape(
    #             test_features, (test_features.shape[0], 4096))

    #         # predict with SVM
    #         clf.fit(train_features, y_train)
    #         pred = clf.predict(test_features)
    #         acc += accuracy_score(y_test, pred)

    #     acc = 100.0 * acc / args.test_number
    #     print('\nTest set: {} way {} shot Accuracy: {:.4f}%'.format(n, k, acc))
    #     return acc

    best_val = 0.0
    test_results = []
    for ep in range(args.epochs):
        training(ep)
        val = validate()
        if val > best_val:
            test_results = []
            # test_results.append(test(5, 1))
            # test_results.append(test(5, 5))
            # test_results.append(test(20, 1))
            # test_results.append(test(20, 5))

    # # Print best results
    # print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
    #     test_results[0]))
    # print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
    #     test_results[1]))
    # print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
    #     test_results[2]))
    # print('\nResult: 5 way 1 shot Accuracy: {:.4f}%\n'.format(
    #     test_results[3]))


if __name__ == '__main__':
    main()
    name1='loss'+'_model:svm.npy'
    name2='acc'+'_model:svm.npy'
    np.save(name1,loss1)
    np.save(name2,acc1)
    draw_loss(loss1,name1)
    draw_acc(acc1,name2)