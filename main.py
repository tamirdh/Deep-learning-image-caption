from PIL.Image import init
from utils.models import CNNtoRNN, get_device
from utils.dataset import get_dataloader, get_dataset
from utils.train import overfit, train
import argparse
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="Full path to the directory containing the images")
    parser.add_argument("annot", type=str,
                        help="Full path to annotations JSON file")
    parser.add_argument("--v_thresh", type=int, default=5,
                        help="Minimal word appearances in order to be added to vocabulary")
    parser.add_argument("--overfit", action='store_true',
                        help="Run a model overfitting on one image")
    parser.add_argument("--load_vocab", action='store_true',
                        help="Use a previously created vocabulary file")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size for dataloader")
    parser.add_argument("T", type=int, help="Number of epochs to run")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    shuffle = not args.overfit
    device = get_device(1)
    dataset = get_dataset(args.data, args.annot,
                          args.v_thresh, args.load_vocab)
    data_loader = get_dataloader(dataset, args.batch, shuffle=shuffle)
    embed_size = 512
    hidden_size = 4096
    vocab_size = len(dataset.vocab)
    model = CNNtoRNN(2048, embed_size, hidden_size, vocab_size)
    if args.overfit:
        overfit(model, device, data_loader, args.T)
    else:
        train(args.T, model, data_loader, device)

    del model
