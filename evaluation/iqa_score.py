import os
import os.path as osp
from datetime import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pyiqa 

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=16, help='Batch size to use')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--device', type=str, default=None, help='Device to use. Like cuda, cuda:0, or cpu')
parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
parser.add_argument('--output-dir', type=str, default='result/', help='Directory to save the output result file')

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'png', 'tiff', 'webp'}

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.image_paths = self._load_images_from_folder(folder_path)

    def _load_images_from_folder(self, folder_path):
        return sorted([
            osp.join(folder_path, f) for f in os.listdir(folder_path)
            if osp.isfile(osp.join(folder_path, f)) and any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_tensor = pyiqa.utils.imread2tensor(img_path)
        # 确保图像形状为 (C, H, W)
        # print(f"Image shape: {img_tensor.shape}")
        return img_tensor

@torch.no_grad()
def calculate_pyiqa_scores(dataloader, model, device):
    total_score = 0.0
    total_images = 0

    for images in tqdm(dataloader, desc="Processing batches"):
        images = images.to(device)
        batch_scores = model(images).cpu().numpy()
        total_score += batch_scores.sum()
        total_images += len(batch_scores)

    avg_score = total_score / total_images if total_images > 0 else 0.0
    return avg_score, total_images

def save_results(output_dir, avg_score, num_images):
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    result_folder = osp.join(output_dir, current_date)
    os.makedirs(result_folder, exist_ok=True)

    result_file = osp.join(result_folder, f"{current_time.replace(':', '-')}_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Time: {current_date} {current_time}\n")
        f.write(f"Task: Image quality\n")
        f.write(f"CLIP IQA: {avg_score:.4f}\n")
        f.write(f"Images: {num_images}\n")
    print(f"Results saved to: {result_file}")

    summary_file = osp.join(output_dir, "summary.txt")
    with open(summary_file, "a") as f:
        f.write(f"[{current_date} {current_time}] CLIP-IQA: {avg_score:.4f} (Images: {num_images})\n")
    print(f"Summary updated: {summary_file}")

def main():
    args = parser.parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    print("Loading PyIQA model...")
    model = pyiqa.create_metric('clipiqa+', device=device).eval()

    print(f"Loading images from: {args.folder_path}")
    dataset = ImageFolderDataset(args.folder_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("Calculating PyIQA average score...")
    avg_score, num_images = calculate_pyiqa_scores(dataloader, model, device)

    save_results(args.output_dir, avg_score, num_images)

if __name__ == "__main__":
    main()


