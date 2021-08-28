import torch.optim as optim
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import torch
from functools import partial
from tqdm import tqdm


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
    plt.imsave(f'{f_name.replace(".png", "")}_{title}.png',img)
    print(f'Saved {f_name} with caption {plt.title}')
    plt.pause(0.001)  # pause a bit so that plots are updated


def train(max_epochs: int, model, data_loader, device: str, progress=250):
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
    # Hyperparameters
    learning_rate = 3e-4
    # init model
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # start epochs
    for _ in range(max_epochs):
        for idx, (img, captions) in tqdm(
            enumerate(data_loader), total=len(data_loader), leave=False
        ):
            img = img.to(device)
            captions = captions.to(device).long()
            output = model(img, captions).to(device)
            loss = criterion(
                output.reshape(-1, output.shape[2]), captions.reshape(-1))
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            if idx > 0 and idx % 250 == 0:
                dataiter = iter(data_loader)
                img_show, cap = next(dataiter)
                output = model(img_show.to(device),
                               cap.to(device).long()).to(device)
                show_img = img_show.to("cpu")
                print(f"\n\nLoss {loss.item():.5f}\n")
                # print(f"\nForward\n")
                out_cap = torch.argmax(output[0], dim=1)
                demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item(
                )] for idx2 in out_cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
                show_image(show_img[0], title=demo_cap, f_name="Forward.png")
                demo_cap = model.caption_images(img_show[0:1].to(
                    device), vocab=data_loader.dataset.vocab, max_len=30)
                demo_cap = ' '.join(demo_cap)
                print("Predicted")
                show_image(img_show[0], title=demo_cap, f_name="Predicted.png")
                print("Original")
                cap = cap[0]
                print(cap.long())
                demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item(
                )] for idx2 in cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
                show_image(img_show[0], title=demo_cap, transform=False, f_name="Original.png")

    return model


def overfit(model,device, data_loader, T=250):
    """
    Run a training on one image+caption
    Args:
        model ([type]): Model to train
        device ([type]): CPU or GPU
        data_loader ([type]): Dataloader
        T (int, optional): How many iterations to run training for. Defaults to 250.
    """
    tqdm_bar = partial(tqdm, position=0, leave=True)

    learning_rate = 3e-4

    
    # init model
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    
    dataiter = iter(data_loader)
    img,caption = next(dataiter)
    for i in tqdm_bar(range(T)):
        # train on the same image and caption to achieve overfitting
        img = img.to(device)
        caption = caption.to(device).long()
        output = model(img, caption).to(device)
        loss = criterion(output.reshape(-1, output.shape[2]), caption.reshape(-1))
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
    
    output = model(img, caption).to(device)
    show_img = img.to("cpu")
    print(f"\n\nLoss {loss.item():.5f}\n")
    out_cap = torch.argmax(output[0],dim=1)
    demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item()] for idx2 in out_cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
    show_image(show_img[0],title=demo_cap, f_name="Forward.png")
    print("Predicted")
    with torch.no_grad():
        model.eval()
        demo_cap = model.caption_images(show_img[0:1].to(device), vocab=data_loader.dataset.vocab, max_len=15)
        demo_cap = ' '.join(demo_cap)
        model.train()
        
        show_image(show_img[0],title=demo_cap, transform=False, f_name="Predicted.png")
    print("Original")
    cap = caption[0]
    #print(cap.long())
    demo_cap = ' '.join([data_loader.dataset.vocab.itos[idx2.item()] for idx2 in cap if idx2.item() != data_loader.dataset.vocab.stoi["<PAD>"]])
    show_image(show_img[0],title=demo_cap, transform=False, f_name="Original.png")