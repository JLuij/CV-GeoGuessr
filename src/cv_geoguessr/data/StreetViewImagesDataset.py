import torch
from torch.utils.data import Dataset

from skimage import io

from cv_geoguessr.data.utils import get_image_file_names_in_dir, take_coordinate_from_file_name
from cv_geoguessr.grid.grid_partitioning import Partitioning


class StreetViewImagesDataset(Dataset):
    def __init__(self, root_dir: str, partitioning: Partitioning, transform=None):
        self.root_dir = root_dir
        self.partitioning = partitioning
        self.transform = transform

        self.all_image_names = get_image_file_names_in_dir(self.root_dir)
        self.all_coordinates = list(
            map(take_coordinate_from_file_name, self.all_image_names))

    def __len__(self):
        return len(self.all_image_names)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: img, label
        """

        if torch.is_tensor(idx):
            # TODO: check out whether this thing really gets called
            print('xxxxx')
            idx = idx.tolist()

        image = io.imread(self.root_dir + '/' + self.all_image_names[idx])

        if self.transform:
            image = self.transform(image)

        label = self.partitioning.one_hot(self.all_coordinates[idx])
        return image, label
