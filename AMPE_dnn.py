import torch.nn as nn
import torch.nn.functional as F


class My_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def main():
    model = My_DNN()
    print(model)

    images, labels = next(iter(valloader))

    img = images[0].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))

    # Ver imagen
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    plt.tight_layout()

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='plasma')
    ax1.axis('off')

    ax2.barh(np.arange(len(ps)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(ps)))
    ax2.set_yticklabels(np.arange(len(ps)))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

     plt.show()



if __name__ == "__main__":
    main()
