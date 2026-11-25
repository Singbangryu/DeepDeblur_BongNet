import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class GoProDataset(Dataset):
    def __init__(
        self,
        set_type,
        data_type="mix",
        dataset_dir="GOPRO_Large",
        debug=False,
        mix_p=0.5,
        patch_size=None,
        augment="augment",
    ):
        """Load paired GoPro images. In mix mode, mix_p is the probability of blur_gamma."""
        if set_type not in {"train", "test"}:
            raise ValueError("set_type must be 'train' or 'test'")
        if data_type not in {"blur", "blur_gamma", "mix"}:
            raise ValueError("data_type must be 'blur', 'blur_gamma' or 'mix'")
        if not 0 <= mix_p <= 1:
            raise ValueError("mix_p must be between 0 and 1")

        self.dataset_dir = dataset_dir
        self.set_dir = Path(dataset_dir) / set_type
        self.data_type = data_type
        self.to_tensor = ToTensor()
        self.debug = debug
        self.mix_p = mix_p
        self.pairs = []
        self.patch_size = patch_size
        self.augment = augment

        folder = "blur" if data_type == "mix" else data_type
        input_paths = sorted(self.set_dir.glob(f"*/{folder}/*.png"))
        if not input_paths:
            raise FileNotFoundError(f"No images found under {self.set_dir}/*/{folder}/")

        for input_path in input_paths:
            scene_dir = input_path.parent.parent
            sharp_path = scene_dir / "sharp" / input_path.name
            if not sharp_path.is_file():
                raise FileNotFoundError(f"Sharp image not found: {sharp_path}")
            self.pairs.append((input_path, sharp_path))

        if debug:
            print(f"Loaded {len(self.pairs)} {data_type} pairs from {self.set_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, sharp_path = self.pairs[index]

        if self.data_type == "mix":
            folder = "blur_gamma" if random.random() < self.mix_p else "blur"
            input_path = sharp_path.parent.parent / folder / sharp_path.name

        with Image.open(input_path) as image:
            input_image = image.convert("RGB")
        with Image.open(sharp_path) as image:
            sharp_image = image.convert("RGB")

        if self.patch_size is not None:
            patch_size = self.patch_size
            width, height = input_image.size
            x = random.randint(0, width - patch_size)
            y = random.randint(0, height - patch_size)
            crop = (x, y, x + patch_size, y + patch_size)
            input_image = input_image.crop(crop)
            sharp_image = sharp_image.crop(crop)

        if self.augment and torch.rand(()).item() < 0.5:
            input_image = input_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            sharp_image = sharp_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        input_tensor = self.to_tensor(input_image)
        sharp_tensor = self.to_tensor(sharp_image)
        if self.debug:
            print(input_tensor.shape, sharp_tensor.shape)
        return input_tensor, sharp_tensor
