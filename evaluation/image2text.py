import os
import os.path as osp
from datetime import datetime
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

from transformers import AutoProcessor, CLIPModel



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model to use (Hugging Face model id)')
parser.add_argument('--num-workers', type=int, help='Number of processes to use for data loading.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use. E.g., cuda, cuda:0, cuda:1, cpu')
parser.add_argument('image_path', type=str, help='Path to the folder containing image files')
parser.add_argument('text_path', type=str, help='Path to the folder containing text files')
parser.add_argument('--output-dir', type=str, default='result/', help='Directory to save the output result file')

class TextImageDataset(Dataset):
    def __init__(self, image_folder, text_folder):
        super().__init__()
        self.text_files = self._get_files(text_folder, extensions=['.txt'])
        self.image_files = self._get_files(image_folder, extensions=['.jpg', '.png', '.jpeg'])
        assert len(self.text_files) == len(self.image_files), "Number of text files and images must be the same"
        self.text_files.sort()
        self.image_files.sort()

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, index):
        text_path = self.text_files[index]
        image_path = self.image_files[index]
        with open(text_path, 'r') as f:
            text = f.read().strip()
        image = Image.open(image_path).convert('RGB')
        # 返回原始文本和PIL图像
        return {'text': text, 'image': image}

    def _get_files(self, folder_path, extensions):
        files = [osp.join(folder_path, f) for f in os.listdir(folder_path)
                 if any(f.lower().endswith(ext) for ext in extensions) and not f.startswith('.')]
        files.sort()
        return files

def custom_collate_fn(batch):
    # batch 是一个包含多条{'text': ..., 'image': ...}记录的列表
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]

    # 返回的仍是字典，但text和image都是列表形式
    return {'text': texts, 'image': images}

@torch.no_grad()
def calculate_clip_t_score(dataloader, clip_model, processor, device):
    clip_score_acc = 0.0
    sample_num = 0.0

    for batch_data in tqdm(dataloader, desc="Processing batches"):
        texts = batch_data['text']   # list of strings
        images = batch_data['image'] # list of PIL Images

        # 使用processor对整批的数据进行预处理
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = clip_model(**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds

        # 归一化特征
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = clip_model.logit_scale.exp()
        clip_score = (logit_scale * (text_features * image_features).sum(dim=1)).mean()

        batch_size = text_features.size(0)
        clip_score_acc += clip_score.item() * batch_size
        sample_num += batch_size

    return clip_score_acc / sample_num if sample_num > 0 else 0.0

def save_result(output_dir, clip_score, clip_model_name, sample_count):
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    result_folder = osp.join(output_dir, current_date)
    os.makedirs(result_folder, exist_ok=True)
    task_type = "Text-to-Image Alignment"
    task_label = "CLIP-T"

    result_file = osp.join(result_folder, f"{current_time.replace(':', '-')}_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Time: {current_date} {current_time}\n")
        f.write(f"Task: {task_type}\n")
        f.write(f"CLIP Model: {clip_model_name}\n")
        f.write(f"{task_label}: {clip_score:.4f}\n")
        f.write(f"Samples: {sample_count}\n")
    print(f"Results saved to: {result_file}")

    summary_file = osp.join(output_dir, "summary.txt")
    with open(summary_file, "a") as f:
        summary_line = (
            f"[{current_date} {current_time}] {task_label}: {clip_score:.4f} Samples: {sample_count}\n"
        )
        f.write(summary_line)
    print(f"Summary updated: {summary_file}")

def main():
    args = parser.parse_args()

    # 设置设备
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    print("Loading CLIP model ")
    processor = AutoProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()

    dataset = TextImageDataset(args.image_path, args.text_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers or 4, collate_fn=custom_collate_fn)

    print("Calculating CLIP-T score...")
    clip_score = calculate_clip_t_score(dataloader, clip_model, processor, device)
    print(f"CLIP-T Score: {clip_score:.4f}")
    save_result(args.output_dir, clip_score, args.clip_model, len(dataset))

if __name__ == "__main__":
    main()
