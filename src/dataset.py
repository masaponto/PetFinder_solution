from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule, LightningModule
from torchvision.io import read_image
import torchvision.transforms as T


class PetfinderDataset(Dataset):
    def __init__(self, df, image_size=224):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        # self._transform = T.Resize([image_size, image_size])
        self._random_crop = T.RandomResizedCrop([image_size, image_size])

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(image_path)
        # image = self._transform(image)
        image = self._random_crop(image)

        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image


class PetfinderDataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

    def __create_dataset(self, train=True):
        return (
            PetfinderDataset(self._train_df, self._cfg.transform.image_size)
            if train
            else PetfinderDataset(self._val_df, self._cfg.transform.image_size)
        )

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, **self._cfg.val_loader)
