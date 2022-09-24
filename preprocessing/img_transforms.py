from torchvision import transforms


class ImageTransform:
    def __init__(self, img_size=64):
        self.transform = {
            "train": transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            "generated": transforms.Compose(
                [
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        img = self.transform[phase](img)

        return img