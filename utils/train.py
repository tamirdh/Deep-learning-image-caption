import pickle
import torch.optim as optim
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import torch
from functools import partial
from tqdm import tqdm
import sys


def show_image(img, title=None, transform=True, f_name=""):
    """Imshow for Tensor."""
    # unnormalize
    if transform:
        img[0] = img[0] * 0.229
        img[1] = img[1] * 0.224
        img[2] = img[2] * 0.225
        img[0] += 0.485
        img[1] += 0.456
        img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))

    # title = title.replace("<SOS>","").replace("<EOS>", "")
    if title is not None:
        plt.title(title)
    if f_name is not None:
        plt.imsave(f'{f_name.replace(".png", "")}_{title}.png', img)
        print(f'Saved {f_name} with caption {plt.title}')
    else:
        plt.imshow(img)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train(max_epochs: int, model, data_loader, device: str, progress=2500):
    """
    Train a given model
    Args:
        max_epochs (int): Number of epoches to train on
        model ([type]): Model to train
        data_loader ([type]): Dataloader
        device (str): CPU or GPU
        progress (int, optional): Show prediction and loss values every X iterations. Defaults to 250.
    Returns:
        [type]: Trained model
    """
    print(f"Using {device}")
    # Monitor
    loss_over_time = list()
    # Hyperparameters
    learning_rate = 1e-4
    
    # init model
    model = model.to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # start epochs
    for epoch in range(max_epochs):
        print(f"Epoch:{epoch}", file=sys.stderr)
        for idx, (img, captions, length) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):  
            
            optimizer.zero_grad()
            img = img.to(device)
            captions = captions.to(device).long()
            output = model(img, captions, length).to(device)
            loss = criterion(output.reshape(-1, output.shape[2]), captions.reshape(-1))
            
            loss.backward()
            optimizer.step()
            if idx > 0 and idx % progress == 0:
                model.eval()
                #torch.save({'model_state_dict': model.state_dict()}, "checkpoint.torch")
                loss_over_time.append(loss.item())
                with open("LOSS.data", "wb") as dest:
                    pickle.dump(loss_over_time, dest)
                with torch.no_grad():
                    output = model(img.to(device), captions.to(device).long(), length).to(device)
                print(f"\n\nLoss {loss.item():.5f}\n")
                print(f"\nForward\n")
                out_cap = torch.argmax(output[0], dim=1)
                demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item(
                )] for idx2 in out_cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
                #show_image(img[0], title=demo_cap, f_name=None)
                print(demo_cap)
                with torch.no_grad():
                    demo_cap = model.caption_images(img[0:1].to(
                        device), vocab=data_loader.dataset.vocab, max_len=30)
                demo_cap = ' '.join(demo_cap)
                print("Predicted")
                print(demo_cap)
                # show_image(img_show[0], title=demo_cap, f_name="Predicted.png")
                print("Original")
                cap = captions[0]
                # print(cap.long())
                demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item(
                )] for idx2 in cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
                print(demo_cap)
                # show_image(img_show[0], title=demo_cap, transform=False, f_name="Original.png")
                sys.stdout.flush()
                model.train()
    return model


def overfit(model, device, data_loader, T=250, img_n = 1):

    """
    Run a training on one image+caption
    Args:
        model ([type]): Model to train
        device ([type]): CPU or GPU
        data_loader ([type]): Dataloader
        T (int, optional): How many iterations to run training for. Defaults to 250.
    """
    assert img_n >= 1, "Use a number larger than 1"
    print(f"Using {device}")
    tqdm_bar = partial(tqdm, position=0, leave=True)

    learning_rate = 3e-4


    # init model
    model = model.to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()


    dataiter = iter(data_loader)
    for _ in range(0, img_n):
        img, caption, length = next(dataiter)
    img = img.to(device)
    caption = caption.to(device).long()
    for i in tqdm_bar(range(T)):
        optimizer.zero_grad()
        # train on the same image and caption to achieve overfitting
        output = model(img, caption, length).to(device)
        loss = criterion(
            output.reshape(-1, output.shape[2]), caption.reshape(-1))
        loss.backward()
        optimizer.step()
        
        print(f"\niteration: {i}")
        print(f"Loss:{loss}")
        print("Predicted:")
        model.eval()
        with torch.no_grad():
            demo_cap = model.caption_images(img[0:1].to(
                device), vocab=data_loader.dataset.vocab, max_len=15)
        demo_cap = ' '.join(demo_cap)
        model.train()
        print(demo_cap)
                

    output = model(img, caption, length).to(device)
    show_img = img.to("cpu")
    print(f"\n\nLoss {loss.item():.5f}\n")
    out_cap = torch.argmax(output[0], dim=1)
    demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item(
    )] for idx2 in out_cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
    print(demo_cap)
    #show_image(show_img[0], title=demo_cap, f_name="Forward.png")
    print("Predicted")
    with torch.no_grad():
        model.eval()
        demo_cap = model.caption_images(show_img[0:1].to(
            device), vocab=data_loader.dataset.vocab, max_len=15)
        demo_cap = ' '.join(demo_cap)
        model.train()
        print(demo_cap)
    #    show_image(show_img[0], title=demo_cap,
    #               transform=False, f_name="Predicted.png")
    print("Original")
    cap = caption[0]
    # print(cap.long())
    demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item(
    )] for idx2 in cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
    #show_image(show_img[0], title=demo_cap,
    #           transform=False, f_name="Original.png")
    print(demo_cap)

def validate_model(model, data_loader, device):
    model = model.to(device)
    model.eval()
    for i in range(10):
        itearator = iter(data_loader)
        img, caption, _ = next(itearator)
        prediction = model.caption_images(img[0:1].to(device), vocab=data_loader.dataset.vocab, max_len=25)
        print(f"\niteration: {i}")
        print(f"Prediction:{' '.join(prediction)}")
        print(f"Original: {caption[0]}") 