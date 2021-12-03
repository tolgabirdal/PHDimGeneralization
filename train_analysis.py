
import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import alexnet, fc, vgg
from topology import calculate_ph_dim
from utils import accuracy, get_data


def get_weights(net):
    with torch.no_grad():
        w = []
        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w)


def eval(eval_loader, net, crit, opt, args, test=True):
    net.eval()

    # run over both test and train set
    with torch.no_grad():    
        total_size = 0
        total_loss = 0
        total_acc = 0
        grads = []
        outputs = []

        P = 0 # num samples / batch size
        for x, y in eval_loader:
            P += 1
            # loop over dataset
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            out = net(x)
            
            outputs.append(out)

            loss = crit(out, y)
            prec = accuracy(out, y)
            bs = x.size(0)

            total_size += int(bs)
            total_loss += float(loss) * bs
            total_acc += float(prec) * bs
        
    hist = [
        total_loss / total_size, 
        total_acc / total_size,
        ]

    print(hist)
    
    return hist, outputs, 0#, noise_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=100, type=int,
        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--dataset', default='mnist', type=str,
        help='mnist | cifar10 | cifar100')
    parser.add_argument('--path', default='data/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='fc', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
        help='NLL | linear_hinge')
    parser.add_argument('--scale', default=64, type=int,
        help='scale of the number of convolutional filters')
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--width', default=100, type=int, 
        help='width of fully connected layers')
    parser.add_argument('--meta_data', default='results', type=str)
    parser.add_argument('--save_file', default='dims.txt', type=str)
    parser.add_argument('--save_ph', default=None)
    parser.add_argument('--save_mst', default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    parser.add_argument('--save_x', default=1000, type=int)
    parser.add_argument('--bn', action='store_true', default=False)
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--ignore_previous', action='store_true', default=False)
    args = parser.parse_args()

    

    # initial setup
    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    print(args)

    # check to see if stuff has run already
    if not args.ignore_previous:
        with open(args.save_file, 'r') as f:
            for line in f.readlines():
                if args.meta_data == line.split(',')[0]:
                    print(f"Metadata {args.meta_data} already ran. Exiting.")
                    exit()


    # training setup
    train_loader, test_loader_eval, train_loader_eval, num_classes = get_data(args)

    if args.model == 'fc':
        if args.dataset == 'mnist':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes).to(args.device)
        elif args.dataset == 'cifar10':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=3*32*32).to(args.device)
    elif args.model == 'alexnet':
        if args.dataset == 'mnist':
            net = alexnet(input_height=28, input_width=28, input_channels=1, num_classes=num_classes)
        else:
            net = alexnet(ch=args.scale, num_classes=num_classes).to(args.device)
    elif args.model == 'vgg':
        net = vgg(depth=args.depth, num_classes=num_classes, batch_norm=args.bn).to(args.device)

    print(net)
    
    opt = getattr(optim, args.optim)(
        net.parameters(), 
        lr=args.lr
        )

    if args.lr_schedule:
        milestone = int(args.iterations / 3)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, 
            milestones=[milestone, 2*milestone],
            gamma=0.5)
    
    crit = nn.CrossEntropyLoss().to(args.device)
    
    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)
    
    # training logs per iteration
    training_history = []

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []

    # weights
    weights_history = deque([])

    STOP = False

    for i, (x, y) in enumerate(circ_train_loader):

        if i % args.eval_freq == 0:
            # first record is at the initial point
            te_hist, te_outputs, te_noise_norm = eval(test_loader_eval, net, crit, opt, args)
            tr_hist, tr_outputs, tr_noise_norm = eval(train_loader_eval, net, crit, opt, args, test=False)
            evaluation_history_TEST.append([i, *te_hist])
            evaluation_history_TRAIN.append([i, *tr_hist])
            if int(tr_hist[1]) == 100:
                print('yaaay all training data is correctly classified!!!')
                STOP = True

        net.train()
        
        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        if torch.isnan(loss):
            print('Loss has gone nan :(.')
            STOP = True

        # calculate the gradients
        loss.backward()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])

        # take the step
        opt.step()

        if i % args.print_freq == 0:
            print(training_history[-1])

        if args.lr_schedule:
            scheduler.step(i)

        if i > args.iterations:
            STOP = True

        weights_history.append(get_weights(net))
        if len(weights_history) > 1000:
            weights_history.popleft()

        # clear cache
        torch.cuda.empty_cache()

        if STOP:
            assert len(weights_history) == 1000

            # final evaluation and saving results
            print('eval time {}'.format(i))
            te_hist, te_outputs, te_noise_norm = eval(test_loader_eval, net, crit, opt, args)
            tr_hist, tr_outputs, tr_noise_norm = eval(train_loader_eval, net, crit, opt, args, test=False)
            evaluation_history_TEST.append([i + 1, *te_hist]) 
            evaluation_history_TRAIN.append([i + 1, *tr_hist])

            weights_history_np = torch.stack(tuple(weights_history)).numpy()
            del weights_history
            ph_dim = calculate_ph_dim(weights_history_np)

            test_acc = evaluation_history_TEST[-1][2]
            train_acc = evaluation_history_TRAIN[-1][2]

            with open(args.save_file, 'a') as f:
                f.write(f"{args.meta_data}, {train_acc}, {test_acc}, {ph_dim}\n")
            
            break
