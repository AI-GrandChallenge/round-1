import os

from torch.utils import data
from torchvision import transforms
from PIL import Image


def get_transform():
    """Module for image pre-processing definition.

    You can customize this module.
    """
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(transform)


class CustomDataset(data.Dataset):
    """Dataset class.

    This class is used for internal NSML inference system. Do not change.
    """
    def __init__(self, root, transform):
        self.data_root = os.path.join(root, 'test_data')
        self.transform = transform
        self.image_ids = [img for img in os.listdir(self.data_root)]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        image = self.transform(image)
        return image, image_id

    def __len__(self):
        return len(self.image_ids)


def data_loader(root, batch_size=64):
    """Test data loading module.

    Args:
        root: string. dataset path.
        batch_size: int.

    Returns:
        DataLoader instance
    """
    input_transform = get_transform()
    dataset = CustomDataset(root, input_transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=False)
