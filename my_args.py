import argparse
 
def build_args_parser():
    parser = argparse.ArgumentParser(description='text classifier')
    parser.add_argument('-modelName',  default='193_spanss', help='')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-dir', type=str, default='out/', help='where to save the snapshot')
    parser.add_argument('-server_addres', type=str, default='http://172.23.131.112:7778', help='where to save the snapshot')
    parser.add_argument('-TaskID', type=str, default='193', help='')
    parser.add_argument('-ClientID', type=str, default='1', help='')
    parser.add_argument('-early-stopping', type=int, default=500,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embedding-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
    parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                        help='comma-separated filter sizes to use for convolution')
    parser.add_argument('-sen_len', type=int, default=1000, help='max length of sentence')
    parser.add_argument('-hidden_size', type=int, default=128, help='max length of sentence')
    parser.add_argument('-hidden_layers', type=int, default=8, help='length of hidden layer count')

    # pre-trained parameters 
    parser.add_argument('-isTrain', type=bool, default=True, help='train or only test')
    parser.add_argument('-public_vocab', type=bool, default=True, help='public_vocab')
    parser.add_argument('-readtype', type=int, default=2, help='public_vocab')
    # device
    parser.add_argument('-device', type=int, default=0,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    # option
    parser.add_argument('-isContinueTrain', type=bool, default=True, help='Continue train from last time')
 
    # dataset
    parser.add_argument('-dataset', type=str, default='data/',
                        help='the path of dataset, ../data/cars_comment/  or ../data/du_query/')
    args = parser.parse_args(args=[])
    return args
