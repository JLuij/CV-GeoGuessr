import torchvision.transforms as transforms
from cv_geoguessr.utils.plot_images import plot_images
from torch.utils.data import DataLoader
from cv_geoguessr.data.StreetViewImagesDataset import StreetViewImagesDataset
import wandb


def get_data_loader(LONDON_PHOTO_DIR, grid_partitioning, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, IMAGENET_MEAN,
                    IMAGENET_STD, brightness = 0.2, contrast = 0.2, saturation = 0.05, hue = 0.1, distortion_scale = .5, p=.4):
    wandb.config.update({"augmentation": {"color_jitter": {"brightness": brightness, "contrast": contrast, "saturation": saturation, "hue": hue }, "random_perspective": {"distortion_scale": distortion_scale, "p": p}}})

    # Add additional random transformation to augment the training dataset
    data_transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomPerspective(distortion_scale = distortion_scale, p = p),
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ColorJitter(brightness = brightness, contrast = contrast, saturation = saturation, hue = hue),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    data_transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_data_set = StreetViewImagesDataset(LONDON_PHOTO_DIR(True), grid_partitioning, data_transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_data_set = StreetViewImagesDataset(LONDON_PHOTO_DIR(False), grid_partitioning, data_transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=TEST_BATCH_SIZE, shuffle=True)

    data_loaders = {
        "train": train_loader,
        "val": test_loader
    }

    data_set_sizes = {
        'train': len(train_data_set),
        'val': len(test_data_set),
    }

    return data_loaders, data_set_sizes


def preview_images(data_loader, SAMPLES_TO_SHOW, IMAGENET_MEAN, IMAGENET_STD, device):
    examples = enumerate(data_loader)
    batch_idx, (eval_images, eval_coordinates) = next(examples)
    eval_images = eval_images.to(device)
    eval_coordinates = eval_coordinates.to(device)

    plot_images(eval_images[:SAMPLES_TO_SHOW].cpu(), IMAGENET_MEAN.cpu(), IMAGENET_STD.cpu())

    return eval_images, eval_coordinates[0, :]


def imagenet_categories():
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    return categories


