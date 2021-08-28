import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import pickle
from vocab import Vocabulary


class COCODataset(Dataset):
    """
    COCODataset
    """

    def __init__(self, root_dir, annotation_file, transform=None, freq_threshold=5,
                 load_vocab=False, vocab_loc="vocab.pkl"):
        """
        can use load_vocab to use a previously created vocabulary (time saving feature)
        freq_threshold: words with a count below this number will be marked as <UNK>
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.cap_max_size = 0
        # Get image and caption colum from the dataframe
        self.imgs = []
        self.captions = []
        for idx, ann in enumerate(self.coco.anns.values()):
            self.imgs.append(self.coco.loadImgs(
                (ann['image_id']))[0]["file_name"])
            self.captions.append(ann['caption'])
            if (idx) % 1000 == 0 and idx > 0:
                print(f"Processed {idx} images and captions")
        print("Finished processing images and captions")
        print(
            f"Got:{len(set(self.imgs))} pictures with {len(self.captions)} captions!")

        # Initialize vocabulary and build vocab
        if load_vocab:
            with open(vocab_loc, "rb") as source:
                self.vocab = pickle.load(source)
            print(f"Loaded vocabulary from {vocab_loc}")

        else:
            print("Build vocabulary")
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions)
            print("Finished building vocabulary")
            with open(vocab_loc, "wb") as dest:
                pickle.dump(self.vocab, dest)

        print(f"Using {len(self.vocab)} words")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transfromation to the image
        if self.transform:
            img = self.transform(img)

        # numericalize the caption text
        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec.extend(self.vocab.numericalize(caption))
        caption_vec.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(caption_vec, dtype=torch.long)


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self, pad_idx, vec_len=-1):
        self.pad_idx = pad_idx
        self.vec_len = vec_len + 2  # adding the <SOS> and <EOS>
        assert self.vec_len > 0, "Vector length must be positive integer"

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets_list = list()
        for item in batch:
            # item = (img:Image, caption:tensor)
            addition = self.vec_len-len(item[1])
            padded_target = torch.cat((item[1], torch.empty(
                addition, dtype=torch.long).fill_(self.pad_idx)), dim=0)
            targets_list.append(padded_target)
            #print(f"GOT:{item[1]}, {item[1].type()}\nAdding:{addition}\nPADDED:{padded_target}\n{padded_target.type()}")
        targets = torch.stack(targets_list, 0)
        return imgs, targets


def get_transformation():
    # define a transformation to add some noise and variance to our images
    transformation = transforms.Compose([transforms.Resize((512, 512), Image.NEAREST),
                                        transforms.ToTensor(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                             0.229, 0.224, 0.225]),
                                         ])
    return transformation


def get_dataset(img_dir, annot_json, threshold=5, load_vocab=False):
    """
    Return a COCODataset object initialised to relevant images and captions

    Args:
        img_dir (str): Images directory path 
        annot_json ([type]): location of annotation.json file
        threshold (int, optional): Minimal number of appearances for a word to be included in the vocabulary. Defaults to 5.
        load_vocab (bool, optional): Load a pre-built vocab.pkl. Defaults to False.

    Returns:
        COCODataset: Used in dataloader to pass images and captions for training
    """
    dataset = COCODataset(
        root_dir=img_dir,
        annotation_file=annot_json,
        transform=get_transformation(),
        freq_threshold=threshold,
        load_vocab=load_vocab
    )
    return dataset

def get_dataloader(dataset: COCODataset, batch_size: int = 4, workers: int= 1, shuffle: bool=False, vec_len:int = 75):
    """
    Returns a PyTorch dataloader using our custom dataset

    Args:
        dataset (COCODataset): Dataset created using COCO
        batch_size (int, optional): Number of images+captions per batch
        workers (int, optional): Refer to https://pytorch.org/docs/stable/data.html
        shuffle (bool, optional): Refer to https://pytorch.org/docs/stable/data.html
        vec_len (int, optional): Length of captions vector
    """
    pad_idx = dataset.vocab.stoi["<PAD>"]

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        collate_fn=CapsCollate(pad_idx=pad_idx, vec_len=vec_len)
    )
    return data_loader
