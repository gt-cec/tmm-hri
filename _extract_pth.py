import torch

if __name__ == "__main__":
    # load the model
    model = torch.load('pytorch_model.bin', map_location='cpu')

    # save the model
    torch.save(model, "./hrnet_ade40k.pth")