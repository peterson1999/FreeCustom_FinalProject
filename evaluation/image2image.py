import os
import os.path as osp
from datetime import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
parser.add_argument('--num-workers', type=int, help='Number of processes to use for data loading.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use. E.g., cuda, cuda:0, cuda:1, cpu')
parser.add_argument('real_path', type=str, help='Path to the folder containing real images')
parser.add_argument('fake_path', type=str, help='Path to the folder containing fake images')
parser.add_argument('--mapping-file', type=str, default=None, help='Optional mapping file for image pairs')
parser.add_argument('--output-dir', type=str, default='result/', help='Directory to save the output result file')

class ImageToImageDataset(Dataset):
    def __init__(self, real_folder, fake_folder, mapping_file=None):
        super().__init__()
        if mapping_file:
            self.data_mapping = self._load_mapping_file(mapping_file, real_folder, fake_folder)
        else:
            self.data_mapping = self._default_mapping(real_folder, fake_folder)

    def __len__(self):
        return len(self.data_mapping)

    def __getitem__(self, index):
        real_path, fake_paths = self.data_mapping[index]
        real_image = self._load_image(real_path)
        fake_images = [self._load_image(fp) for fp in fake_paths]
        return {'real': real_image, 'fake_list': fake_images}

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _default_mapping(self, real_folder, fake_folder):
        real_files = self._get_files(real_folder)
        fake_files = self._get_files(fake_folder)
        real_files.sort()
        fake_files.sort()
        assert len(real_files) == len(fake_files), "Number of real and fake images must be the same"
        return [(r, [f]) for r, f in zip(real_files, fake_files)]  # 1-to-1 mapping

    def _load_mapping_file(self, mapping_file, real_folder, fake_folder):
        data_mapping = []
        with open(mapping_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                real_img_path = osp.join(real_folder, parts[0])
                fake_img_paths = [osp.join(fake_folder, p.strip()) for p in parts[1:]]
                data_mapping.append((real_img_path, fake_img_paths))
        return data_mapping

    def _get_files(self, folder_path):
        files = [osp.join(folder_path, f) for f in os.listdir(folder_path)
                 if not f.startswith('.') and f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        return files

def custom_collate_fn(batch):
    real_images = [item['real'] for item in batch]
    fake_images_list = [item['fake_list'] for item in batch]

    # Flatten all fake images and keep track of counts per real image
    all_fake_images = []
    fake_counts = []
    for fake_list in fake_images_list:
        all_fake_images.extend(fake_list)
        fake_counts.append(len(fake_list))

    fake_counts = torch.tensor(fake_counts)
    return {'real_images': real_images, 'fake_images': all_fake_images, 'fake_counts': fake_counts}

@torch.no_grad()
def calculate_scores(dataloader, clip_model, dino_model, device, clip_processor, dino_processor):
    cos = nn.CosineSimilarity(dim=-1)
    clip_score_acc = 0.0
    dino_score_acc = 0.0
    num_real_images = 0

    for batch_data in tqdm(dataloader, desc="Processing batches"):
        real_images_pil = batch_data['real_images']
        fake_images_pil = batch_data['fake_images']
        fake_counts = batch_data['fake_counts']

        # 准备CLIP输入
        # 注意：clip_processor一次可以处理多张图像，这里统一处理
        real_inputs_clip = clip_processor(images=real_images_pil, return_tensors="pt").to(device)
        fake_inputs_clip = clip_processor(images=fake_images_pil, return_tensors="pt").to(device)

        # 提取CLIP特征
        real_features_clip = clip_model.get_image_features(**real_inputs_clip)
        fake_features_clip = clip_model.get_image_features(**fake_inputs_clip)

        # 归一化
        real_features_clip = real_features_clip / real_features_clip.norm(dim=-1, keepdim=True)
        fake_features_clip = fake_features_clip / fake_features_clip.norm(dim=-1, keepdim=True)

        # DINO 输入
        real_inputs_dino = dino_processor(images=real_images_pil, return_tensors='pt').to(device)
        fake_inputs_dino = dino_processor(images=fake_images_pil, return_tensors='pt').to(device)

        real_outputs_dino = dino_model(**real_inputs_dino).last_hidden_state
        fake_outputs_dino = dino_model(**fake_inputs_dino).last_hidden_state

        # 对DINO特征求平均
        real_features_dino = real_outputs_dino.mean(dim=1)
        fake_features_dino = fake_outputs_dino.mean(dim=1)

        # 归一化DINO特征
        real_features_dino = real_features_dino / real_features_dino.norm(dim=-1, keepdim=True)
        fake_features_dino = fake_features_dino / fake_features_dino.norm(dim=-1, keepdim=True)

        idx = 0
        batch_size = len(real_images_pil)
        num_real_images += batch_size

        for i, count in enumerate(fake_counts):
            real_clip_feat = real_features_clip[i].unsqueeze(0)
            fake_clip_feats = fake_features_clip[idx:idx+count]
            clip_sims = (real_clip_feat * fake_clip_feats).sum(dim=1).mean().item()

            real_dino_feat = real_features_dino[i].unsqueeze(0)
            fake_dino_feats = fake_features_dino[idx:idx+count]
            dino_sims = (real_dino_feat * fake_dino_feats).sum(dim=1).mean().item()

            clip_score_acc += clip_sims
            dino_score_acc += dino_sims
            idx += count

    average_clip_score = clip_score_acc / num_real_images if num_real_images > 0 else 0.0
    average_dino_score = dino_score_acc / num_real_images if num_real_images > 0 else 0.0
    return average_clip_score, average_dino_score

def save_result(output_dir, clip_score, dino_score, sample_count):
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    result_folder = osp.join(output_dir, current_date)
    os.makedirs(result_folder, exist_ok=True)
    task_type = "Image-to-Image Similarity"
    task_label = "CLIP-I"

    result_file = osp.join(result_folder, f"{current_time.replace(':', '-')}_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Time: {current_date} {current_time}\n")
        f.write(f"Task: {task_type}\n")
        f.write(f"CLIP Model: ViT-B/32\n")
        f.write(f"DINO Model: dinov2_vitb14\n")
        f.write(f"{task_label}: {clip_score:.4f}, DINO: {dino_score:.4f}\n")
        f.write(f"Samples: {sample_count}\n")
    print(f"Results saved to: {result_file}")

    summary_file = osp.join(output_dir, "summary.txt")
    with open(summary_file, "a") as f:
        summary_line = (
            f"[{current_date} {current_time}] {task_label}: {clip_score:.4f}, "
            f"DINO: {dino_score:.4f} Samples: {sample_count}\n"
        )
        f.write(summary_line)
    print(f"Summary updated: {summary_file}")

def main():
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    print(f"Loading CLIP model")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()

    print("Loading DINOv2 model ")
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    dino_model.eval()

    dataset = ImageToImageDataset(args.real_path, args.fake_path, args.mapping_file)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn,
        num_workers=args.num_workers or 4
    )
    print("Calculating scores...")
    clip_score, dino_score = calculate_scores(
        dataloader, clip_model, dino_model, device, clip_processor, dino_processor
    )
    print(f"CLIP-I Score: {clip_score:.4f}")
    print(f"DINOv2 Score: {dino_score:.4f}")
    save_result(args.output_dir, clip_score, dino_score, len(dataset))

if __name__ == "__main__":
    main()
