import os
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

class FileDataset(IterableDataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []

    def _get_image_paths(self):
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.jpg'):
                        yield os.path.join(folder_path, file_name)

    def __iter__(self):
        for img_path in self._get_image_paths():
            yield read_image(img_path)

if __name__ == "__main__":
    data_directory = 'images'
    dataset = FileDataset(root_dir=data_directory)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    for batch in dataloader:
        print(batch.shape)

        img = transforms.ToPILImage()(batch[0])

        img.save('cruft/output_image.png')
        break