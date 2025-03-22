from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm


def combine_file(folder_path):
    dfs = []
    first_file = True
    column_names = None

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if first_file:
                column_names = [col.strip()
                                for col in lines[0].strip().split('\t')]
                first_file = False

            data = [line.strip().split('\t') for line in lines[1:]]
            df = pd.DataFrame(data, columns=column_names)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip()
    combined_df = combined_df.applymap(
        lambda x: x.strip() if isinstance(x, str) else x)

    return combined_df


def categorize_age(age):
    try:
        if 'None' in age:
            return 'Unknown'
        elif isinstance(age, str):
            age = age.strip()
            if age.startswith('(') and age.endswith(')'):
                lower, upper = map(int, age[1:-1].split(', '))
                age = (lower + upper) // 2
            else:
                age = int(age)
    except Exception as e:
        raise ValueError(f"Unable to parse age value: {age}") from e

    if age < 7:
        return 'Infant (0-6 years)'
    elif age < 13:
        return 'Child (7-12 years)'
    elif age < 21:
        return 'Teenager (13-20 years)'
    elif age < 36:
        return 'Young Adult (21-35 years)'
    elif age < 56:
        return 'Adult (36-55 years)'
    elif age >= 56:
        return 'Senior (56 years and above)'

    raise ValueError(f"Unable to categorize age value: {age}")


def get_fold_data_df(folder_path):
    fold_data_df = combine_file(folder_path)
    fold_data_df['age_group'] = fold_data_df['age'].apply(categorize_age)
    print(fold_data_df['age_group'].value_counts())
    fold_data_df['image_route'] = fold_data_df.apply(
        lambda row: folder_path + 'aligned/' + row['user_id'] + '/' + 'landmark_aligned_face.' + row['face_id'] + '.' + row['original_image'], axis=1)
    fold_data_df = fold_data_df[['image_route', 'age_group']]
    print(fold_data_df.head())
    return fold_data_df


def get_adience_train_test_loader(batchsize):
    folder_path = 'data/AdienceGender/'
    fold_data_df = get_fold_data_df(folder_path)
    fold_data_df = fold_data_df[fold_data_df['age_group'] != 'Unknown']

    transform = Compose([
        Resize(224),
        transforms.ToTensor(),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
    ])

    age_groups = sorted(fold_data_df['age_group'].unique())
    label_map = {age: idx for idx, age in enumerate(age_groups)}

    train_df, test_df = train_test_split(
        fold_data_df, test_size=0.2, stratify=fold_data_df['age_group'], random_state=42
    )

    def image_loader(img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            return transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

    train_images = [image_loader(path) for path in tqdm(
        train_df['image_route'], total=len(train_df), desc='Loading training images...')]
    train_labels = [label_map[label] for label in train_df['age_group']]

    test_images = [image_loader(path) for path in tqdm(
        test_df['image_route'], total=len(test_df), desc='Loading test images...')]
    test_labels = [label_map[label] for label in test_df['age_group']]

    train_images = torch.stack(train_images)
    test_images = torch.stack(test_images)

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_adience_train_test_loader(batchsize=HP.batch_size)
