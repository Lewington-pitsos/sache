import os
import hashlib
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

class FileDataset(IterableDataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _get_image_paths(self):
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.jpg'):
                        yield os.path.join(folder_path, file_name)

    def _worker_iter(self, worker_id, num_workers):
        for img_path in self._get_image_paths():
            # Compute a deterministic hash of the image path
            hash_val = int(hashlib.sha1(img_path.encode('utf-8')).hexdigest(), 16)
            # Assign image to worker based on hash value
            if hash_val % num_workers == worker_id:
                yield read_image(img_path, mode=ImageReadMode.RGB)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for img_path in self._get_image_paths():
                yield read_image(img_path, mode=ImageReadMode.RGB)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            yield from self._worker_iter(worker_id, num_workers)


class FilePathDataset(FileDataset):
    def __iter__(self):
        for img_path in self._get_image_paths():
            yield img_path, read_image(img_path, mode=ImageReadMode.RGB)

if __name__ == "__main__":
    data_directory = 'images'
    dataset = FileDataset(root_dir=data_directory)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
    for batch in dataloader:
        print(batch.shape)

        img = transforms.ToPILImage()(batch[0])

        img.save('cruft/output_image.png')
        break