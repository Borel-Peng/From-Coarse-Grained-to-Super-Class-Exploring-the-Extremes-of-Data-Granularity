from torch.utils.data import Dataset
import shutil
from collections import defaultdict
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import requests
import os
from flickrapi import FlickrAPI
from tqdm import tqdm
import time
from pathlib import Path
import sys
import numpy as np
from torchvision.datasets import ImageFolder


class DownloadProgress:
    def __init__(self):
        self.total_photos = 0
        self.successful_downloads = 0
        self.failed_downloads = 0

    def print_stats(self):
        print(f"\nProgress Statistics:")
        print(f"Total photos processed: {self.total_photos}")
        print(f"Successful downloads: {self.successful_downloads}")
        print(f"Failed downloads: {self.failed_downloads}")
        if self.total_photos > 0:
            success_rate = (self.successful_downloads /
                            self.total_photos) * 100
            print(f"Success rate: {success_rate:.2f}%")


def test_flickr_connection():
    print("Testing Flickr API connection...")
    try:
        flickr = FlickrAPI(FLICKR_API_KEY, FLICKR_API_SECRET,
                           format='parsed-json')
        flickr.photos.getRecent(per_page=1)
        print("Successfully connected to Flickr API!")
        return flickr
    except Exception as e:
        print(f"Failed to connect to Flickr API: {str(e)}")
        sys.exit(1)


def get_photo_info(flickr, photo_id):
    try:
        photo_info = flickr.photos.getInfo(photo_id=photo_id)
        return photo_info
    except Exception as e:
        print(f"Error getting info for photo {photo_id}: {str(e)}")
        return None


def get_photo_url(photo_info):
    try:
        photo = photo_info['photo']
        farm_id = photo['farm']
        server_id = photo['server']
        photo_id = photo['id']
        secret = photo['secret']

        url = f"https://farm{farm_id}.staticflickr.com/{server_id}/{photo_id}_{secret}.jpg"
        return url
    except Exception as e:
        print(f"Error constructing URL: {str(e)}")
        return None


def download_photo(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Network error while downloading: {str(e)}")
        return False
    except Exception as e:
        print(f"Error saving photo: {str(e)}")
        return False


def download():
    progress = DownloadProgress()
    flickr = test_flickr_connection()
    print("\nCreating directories...")
    base_dir = Path('flickr_photos')
    for category in ['people', 'animals', 'urban', 'nature']:
        (base_dir / category).mkdir(parents=True, exist_ok=True)
    print("Directories created successfully!")

    print("\nReading dataset...")
    try:
        df = pd.read_csv('beauty-icwsm15-dataset.tsv', sep='\t')
        print(f"Successfully loaded dataset with {len(df)} entries")
        print(f"Columns: {df.columns}")
    except Exception as e:
        print(f"Error reading dataset: {str(e)}")
        return

    print("\nStarting download process...")
    for index, row in df.iterrows():
        progress.total_photos += 1
        photo_id = str(row['#flickr_photo_id'])
        category = row['category']

        print(f"\nProcessing photo {index+1}/{len(df)}")
        print(f"Photo ID: {photo_id}, Category: {category}")

        save_path = base_dir / category / f"{photo_id}.jpg"

        if save_path.exists():
            print(f"Photo {photo_id} already exists, skipping...")
            progress.successful_downloads += 1
            continue

        photo_info = get_photo_info(flickr, photo_id)
        if not photo_info:
            print(f"Failed to get info for photo {photo_id}, skipping...")
            progress.failed_downloads += 1
            continue

        url = get_photo_url(photo_info)
        if not url:
            print(f"Failed to construct URL for photo {photo_id}, skipping...")
            progress.failed_downloads += 1
            continue

        print(f"Downloading photo {photo_id}...")
        success = download_photo(url, save_path)

        if success:
            print(f"Successfully downloaded photo {photo_id}")
            progress.successful_downloads += 1
        else:
            print(f"Failed to download photo {photo_id}")
            progress.failed_downloads += 1

        time.sleep(1)

        if index % 10 == 0:
            progress.print_stats()

    print("\nDownload process completed!")
    progress.print_stats()


class FlickrBeautyDataset(Dataset):
    def __init__(self, tsv_file, img_dir, transform=None, rewrite=False, output_dir=None):
        self.data = pd.read_csv(tsv_file, sep='\t')
        self.img_dir = img_dir
        self.transform = transform
        self.rewrite = rewrite
        self.output_dir = output_dir

        self.data['median_score'] = self.data['beauty_scores'].apply(
            lambda x: int(np.median([int(score)
                          for score in x.split(',')]) - 1)
        )

        self.image_files = []
        for category in os.listdir(img_dir):
            category_path = os.path.join(img_dir, category)
            if os.path.isdir(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith(".jpg"):
                        img_id = filename.split('.')[0]
                        self.image_files.append((category, img_id))

        self.data_filtered = self.data[self.data['#flickr_photo_id'].isin(
            [int(img_id) for _, img_id in self.image_files])]

        if self.rewrite and self.output_dir:
            self._rewrite_to_local()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        category, img_id = self.image_files[idx]

        data_info = self.data_filtered[self.data_filtered['#flickr_photo_id'] == int(
            img_id)].iloc[0]

        img_path = os.path.join(self.img_dir, category, f"{img_id}.jpg")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        score = data_info['median_score']

        return image, torch.tensor(score, dtype=torch.long)

    def _rewrite_to_local(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for score in self.data_filtered['median_score'].unique():
            score_dir = os.path.join(self.output_dir, f'score_{score}')
            if not os.path.exists(score_dir):
                os.makedirs(score_dir)

        for _, img_id in self.image_files:
            category = self.data_filtered[self.data_filtered['#flickr_photo_id'] == int(
                img_id)].iloc[0]['category']
            score = self.data_filtered[self.data_filtered['#flickr_photo_id'] == int(
                img_id)].iloc[0]['median_score']
            img_path = os.path.join(self.img_dir, category, f"{img_id}.jpg")
            new_img_path = os.path.join(
                self.output_dir, f'score_{score}', f"{img_id}.jpg")
            shutil.copy(img_path, new_img_path)


def get_aug_train_test_loader(batchsize):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409], [
                             0.2009, 0.1984, 0.2023])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409],
                             [0.2009, 0.1984, 0.2023])
    ])

    dataset = FlickrBeautyDataset(
        tsv_file='data/flickr_data/beauty-icwsm15-dataset.tsv',
        img_dir='data/flickr_data',
        transform=None
    )

    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_images = []
    train_labels = []
    to_pil = transforms.ToPILImage()

    for img, label in tqdm(train_dataset, desc="Processing train data"):
        pos_1 = train_transform(img)
        pos_2 = train_transform(img)
        train_images.append(pos_1)
        train_images.append(pos_2)
        train_labels.append(label)
        train_labels.append(label)

    train_images_tensor = torch.stack(train_images)
    train_labels_tensor = torch.tensor(train_labels)

    train_loader = DataLoader(
        TensorDataset(train_images_tensor, train_labels_tensor),
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    test_images = []
    test_labels = []
    for img, label in tqdm(test_dataset, desc="Processing test data"):
        img = test_transform(img)
        test_images.append(img)
        test_labels.append(label)

    test_images_tensor = torch.stack(test_images)
    test_labels_tensor = torch.tensor(test_labels)

    test_loader = DataLoader(
        TensorDataset(test_images_tensor, test_labels_tensor),
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    return train_loader, test_loader


def get_train_test_loader(batchsize):
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409], [
                             0.2009, 0.1984, 0.2023])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409],
                             [0.2009, 0.1984, 0.2023])
    ])

    dataset = FlickrBeautyDataset(
        tsv_file='data/flickr_data/beauty-icwsm15-dataset.tsv',
        img_dir='data/flickr_data',
        transform=None
    )

    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_images = []
    train_labels = []
    to_pil = transforms.ToPILImage()

    for img, label in tqdm(train_dataset, desc="Processing train data"):
        pos_1 = train_transform(img)
        train_images.append(pos_1)
        train_labels.append(label)

    train_images_tensor = torch.stack(train_images)
    train_labels_tensor = torch.tensor(train_labels)

    train_loader = DataLoader(
        TensorDataset(train_images_tensor, train_labels_tensor),
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    test_images = []
    test_labels = []
    for img, label in tqdm(test_dataset, desc="Processing test data"):
        img = test_transform(img)
        test_images.append(img)
        test_labels.append(label)

    test_images_tensor = torch.stack(test_images)
    test_labels_tensor = torch.tensor(test_labels)

    test_loader = DataLoader(
        TensorDataset(test_images_tensor, test_labels_tensor),
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    return train_loader, test_loader


def get_original_label_train_test_loader(batchsize):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4866, 0.4409],
                             [0.2009, 0.1984, 0.2023])
    ])

    dataset = ImageFolder(
        root='data/flickr_data', transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    tsv_file = 'data/flickr_data/beauty-icwsm15-dataset.tsv'
    img_dir = 'data/flickr_data'
    output_dir = 'data_flickr'
    rewrite = True

    dataset = FlickrBeautyDataset(tsv_file=tsv_file,
                                  img_dir=img_dir,
                                  transform=None,
                                  rewrite=rewrite,
                                  output_dir=output_dir)
