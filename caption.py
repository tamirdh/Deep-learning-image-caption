from utils.models import CNNtoRNN, get_device
from utils.train import show_image
import torchvision.transforms as transforms
from PIL import Image
import torch
import pickle
import string
import argparse


class Captioner:
    def __init__(self, weight_path:str, vocab_path:str) -> None:
        self.transformation = self._get_image_transformation()
        self.device = get_device(1)
        with open(vocab_path, "rb") as source:
            self.vocab = pickle.load(source)
        
        self.model = CNNtoRNN(embed_size=512, hidden_size=2096,vocab_size=len(self.vocab), n_features=0)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device)["model_state_dict"])
        self.model.eval()


    def _get_image_transformation(self):
        """Transform any image to work with the pre-trained CNN 

        Returns:
            transformation function
        """
        transformation = transforms.Compose([transforms.Resize((299, 299), Image.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                                0.229, 0.224, 0.225]),
                                            ])
        return transformation



    def _caption_image(self, image:torch.Tensor):
        caption = self.model.caption_image(image, self.vocab, 80)
        caption_text = ' '.join(caption)
        caption_text = caption_text.replace("<SOS>", "").replace("<EOS>", "").replace("<PAD>", "")
        for symbol in string.punctuation:
            caption_text = caption_text.replace(f" {symbol}", symbol)
        return caption_text
    
    def get_caption(self, img_path:str):
        img = Image.open(img_path).convert("RGB")
        img = self.transformation(img).to(self.device).unsqueeze(0)
        return self._caption_image(img)
    
    def show_caption(self, img_path:str, save:bool=False):
        img = Image.open(img_path).convert("RGB")
        img = self.transformation(img).to(self.device).unsqueeze(0)
        caption = self._caption_image(img)
        if save:
            show_image(img.cpu().squeeze(0), caption, f_name=f"imgs/{caption}.png")
        else:
            show_image(img.cpu().squeeze(0), caption, f_name=None)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_weights.pt", help="Path to trained model weights file")
    parser.add_argument("--vocab", default="vocab.pkl", help="Path to vocabulary that was used during training")
    parser.add_argument("--img", help="Path to an image file that will be captioned")
    parser.add_argument("--no-gui", help="Disables image show and only outputs the caption", action='store_true', dest="gui")
    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    if args.img is None:
        print("No image was supplied")
    else:
        captioner = Captioner(args.model, args.vocab)
        if args.gui:
            captioner.show_caption(args.img, save=True)
        else:
            captioner.show_caption(args.img)

        