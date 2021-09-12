from PIL.Image import init
import torch.optim as optim
from utils.models import CNNtoRNN, get_device
from utils.dataset import get_dataloader, get_dataset
from utils.train import overfit, train, validate_model
import argparse
import os
import torch

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
    parser.add_argument("--progress", type=int, default=250, help="Show training loss every X iterations")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="path to model's checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    return parser.parse_args()



if __name__ == '__main__':
    args = init_args()
    print(args)
    shuffle = not args.overfit
    print(f"Shuffling dataset:{shuffle}")
    device = get_device(1)
    dataset = get_dataset(args.data, args.annot,
                          args.v_thresh, args.load_vocab)
    data_loader = get_dataloader(dataset, args.batch, shuffle=shuffle)
    vocab_size = len(dataset.vocab)
    embed_size = 512
    hidden_size = 2096
    learning_rate = 1e-4
    n_features = 0 # used for old RNNs
    print(f"Embed size:{embed_size}\nHidden size:{hidden_size}")
    
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, n_features)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # ADD WHEN RESUME WORK ON SAME MODEL
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    print(model)
    if args.overfit:
        overfit(model, device, data_loader, args.T, 2)
        # model.load_state_dict(torch.load("checkpoint.torch")["model_state_dict"])
        #validate_model(model, data_loader, device)

    else:
        train(args.T, model.train(), optimizer, data_loader, device, args.checkpoint, args.progress)
    
    del model
