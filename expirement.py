from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from torch.utils.data import dataloader
from utils.dataset import get_dataloader, get_dataset
from utils.train import train
from utils.models import *
import os
import torch
import pickle
import json
import argparse


class Expirement:
    def __init__(self, mode: str = "train", use_train: bool = False, epochs: int = 3, model: CNNtoRNN = None,
                 load_model: bool = False, username: str = None, cnn:str=None, rnn:str=None ):
        self.use_train = use_train
        self.mode = mode
        self.epochs = epochs
        self.user = username
        self._fig = self._get_plot()
        self.device = get_device(1)
        
        self._loss_prg = 1000
        # Makes sure that all expirements are using the same vocabulary
        self.dataset = get_dataset(
            self._get_img_dir(), self._get_annot_file(), threshold=5, load_vocab=True)
        batch_size = 4 if mode == "train" else 1
        self.dataloader = get_dataloader(
            self.dataset, batch_size, shuffle=True)

        self.model = fix_model(model, rnn, cnn).to(self.device)
        if not os.path.exists("results"):
            os.makedirs("results")
        self.fname = f"{self.model.encoderCNN.__class__.__name__}_{self.model.decoderRNN.__class__.__name__}_{self.model.decoderRNN.hidden_size}_{self.model.decoderRNN.embed_size}"
        if load_model:
            self.model.load_state_dict(
                torch.load(f"results/model_{self.fname}.data")["model_state_dict"])
            self.model.eval()
        print(f"Starting Expirement with {self.fname}")
        print(f"Using train set: {self.use_train}")
        print(f"Batch size: {batch_size}")

    def _get_img_dir(self):
        if self.use_train:
            img_dir = f"/home/yandex/DLW2021/{self.user}/coco/train2017"
        else:
            img_dir = f"/home/yandex/DLW2021/{self.user}/coco/val2017"
        return img_dir

    def _get_annot_file(self):
        if self.use_train:
            annot_file = f"/home/yandex/DLW2021/{self.user}/coco/annotations/captions_train2017.json"
        else:
            annot_file = f"/home/yandex/DLW2021/{self.user}/coco/annotations/captions_val2017.json"
        return annot_file

    def _get_plot(self):
        figure = plt.figure(figsize=(12, 12))
        return figure

    def plot_multiple_losses(self):
        files = [file for file in os.listdir(
            "results") if file.startswith("LOSS")]
        ax = self._fig.add_subplot(111)
        for file in files:
            with open(file, "rb") as source:
                file_data = pickle.load(source)
            x = [i*self._loss_prg for i in range(1, len(file_data)+1)]
            label = os.path.basename(file).replace(
                "LOSS_", "").replace(".data", "")
            ax.plot(x, file_data, label=label)
        ax.set_title("Loss over training")
        ax.set_xlabel("Iteration number")
        ax.set_ylabel("Loss Value")
        ax.set_xscale('log')
        ax.legend(loc="upper right")
        plt.savefig('loss_plot.png')
        print("Loss Plot Saved")

    def run_train_expirement(self):
        """
        Train the model and create the relevant LOSS data file
        """
        self.model = train(self.epochs, self.model,
                           self.dataloader, self.device, self._loss_prg, self.fname)

    def run_validation_expirement(self):
        """
        Run the model on the validation set and show average BLEU-N scores for N=1,2,3,4
        """
        self.model.eval()
        bleu_1 = 0
        bleu_2 = 0
        bleu_3 = 0
        bleu_4 = 0
        dataiter = iter(self.dataloader)
        for idx in range(10):
            img, caption, _ = next(dataiter)
            img = img.to(self.device)
            caption = caption.to(self.device)
            real_cap = [self.dataset.vocab.itos[idx2.item(
            )] for idx2 in caption[0] if idx2.item() != self.dataset.vocab.stoi["<PAD>"]]
            with torch.no_grad():
                hypothesis = self.model.caption_image(
                    img, self.dataset.vocab, max_len=77)
            bleu_1 += sentence_bleu(real_cap, hypothesis, weights=(1, 0, 0, 0))
            bleu_2 += sentence_bleu(real_cap, hypothesis,
                                    weights=(0.5, 0.5, 0, 0))
            bleu_3 += sentence_bleu(real_cap, hypothesis,
                                    weights=(0.33, 0.33, 0.33, 0))
            bleu_4 += sentence_bleu(real_cap, hypothesis,
                                    weights=(0.25, 0.25, 0.25, 0.25))
            if ((idx+1) % 100) == 0:
                print(f"Validated {idx+1} images")
        bleu_1 = bleu_1/len(self.dataloader)
        bleu_2 = bleu_2/len(self.dataloader)
        bleu_3 = bleu_3/len(self.dataloader)
        bleu_4 = bleu_4/len(self.dataloader)
        result = {"bleu-1": bleu_1, "bleu-2": bleu_2,
                  "bleu-3": bleu_3, "bleu-4": bleu_4}
        fname = f"results/bleu_{self.fname}.data"
        with open(fname, "wb") as dest:
            pickle.dump(result, dest)
        print("Finished BLEU evaluation")
        print(json.dumps(result, indent=4))

    def plot_bleu_results(self):
        bleu_files = [file for file in os.listdir(
            "results") if file.startswith("bleu")]
        col_label = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
        row_label = list()
        cell_text = list()
        for file in bleu_files:
            with open(f"results/{file}", "rb") as source:
                bleu_dict = pickle.load(source)
            cell_text.append([bleu_dict["bleu-1"], bleu_dict["bleu-2"],
                              bleu_dict["bleu-3"], bleu_dict["bleu-4"]])
            row_label.append(file.replace("bleu_", "").replace(".data", ""))
        fig, ax = plt.subplots(figsize=(25, 6))
        ax.set_axis_off()
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_label,
            colLabels=col_label,
            cellLoc='center',
            loc='upper left',
        )

        ax.set_title("BLEU-N Score of Models", fontweight="bold")
        plt.savefig('bleu_table.png', dpi=1200, format="png")

    def full_expirement(self):
        self.model.train()
        self.run_train_expirement()
        self.run_validation_expirement()


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str,
                        help="Username for coco images path", required=True)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs in training", required=False)
    parser.add_argument("--hidden", type=int, default=2096,
                        help="Hidden layer size", required=True)
    parser.add_argument("--embed", type=int, default=512,
                        help="Embedding size", required=True)
    parser.add_argument(
        "--cnn", choices=["v1", "v2"], help="CNN encoder version", required=True)
    parser.add_argument("--rnn", choices=["v1", "v2", "v3", "v4",
                                          "v5", "greedy"], help="RNN Decoder version", required=True)
    parser.add_argument("--use-train", action="store_true", help="Use train of val of COCO", dest="use_train")
    parser.add_argument("--plot", action="store_true", help="Don't run any expirement, just plot")
    return parser.parse_args()


def fix_model(model:CNNtoRNN, rnn, cnn, dataset):
    vocab_size = len(dataset.vocab)
    if cnn == "v1":
        cnn = EncoderCNN(embed, False)
    else:
        cnn = EncoderCNNV2(embed)
    if rnn == "v1":
        rnn = DecoderRNN(embed, hidden_size=hidden, vocab_size=vocab_size, n_features=embed)
    elif rnn == "v2":
        rnn = DecoderRNNV2(embed, hidden, vocab_size, embed)
    elif rnn == "v3":
        rnn = DecoderRNNV3(embed, hidden, vocab_size, embed)
    elif rnn == "v4": 
        rnn = DecoderRNNV4(embed, hidden, vocab_size, None)
    elif rnn == "v5": 
        rnn = DecoderRNNV5(embed, hidden, vocab_size, None)
    else:
        rnn = DecoderRNNEGreed(embed, hidden, vocab_size, embed)
    
    model.encoderCNN = cnn
    model.decoderRNN = rnn
    return model

if __name__ == '__main__':
    args = init_args()
    model = CNNtoRNN(0, 0, 0, 0)
    hidden = args.hidden
    embed = args.embed
        
    exp = Expirement("train", False, args.epochs, model, False, args.username, args.cnn, args.rnn)
    if args.plot:
        exp.plot_bleu_results()
        exp.plot_multiple_losses()
    else:
        # Create a new model and train it. Records loss over train.
        exp.run_train_expirement()
        # New expirement, loads previously trained model and validate.
        exp = Expirement("eval", False, args.epochs, model, True, args.username, args.cnn, args.rnn)
        exp.run_validation_expirement()
