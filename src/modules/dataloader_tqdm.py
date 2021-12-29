import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

transforms_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])

dataset_test = CIFAR10(root='../data', train=False, download=True, transform=transforms_test)
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=4)

progress_bar = tqdm(test_loader)
for i,(image,labels) in enumerate(progress_bar):
    ii = 0