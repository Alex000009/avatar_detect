import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor

class ResizeAndPadWithBoxes:
    def __init__(self, size):
        self.size = size
        self.to_tensor = ToTensor()

    def __call__(self, img, target):
        target_h, target_w = self.size
        w, h = img.size

        # Compute scale factor
        scale = min(target_w / w, target_h / h)

        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)

        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        # Pad image
        img = TF.pad(img, (left, top, right, bottom), fill=0)

        # Adjust boxes
        boxes = target["boxes"]
        boxes = boxes * scale
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top
        target["boxes"] = boxes

        # Convert to tensor
        img = self.to_tensor(img)

        return img, target

class CocoDetectionTransform(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self._transform = transform

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # Convert target to format expected by Faster R-CNN
        boxes = []
        labels = []
        for obj in target:
            # Get bounding box in [xmin, ymin, width, height] format
            bbox = obj['bbox']
            # Convert to [xmin, ymin, xmax, ymax]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            
            # Skip invalid boxes (width or height <= 0)
            if xmax <= xmin or ymax <= ymin:
                continue
                
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])
            
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self._transform is not None:
            img, target = self._transform(img, target)
            
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    root = r"D:\Desktop\AI\project\images\train"
    ann = r"D:\Desktop\AI\project\annotations\instances_train.json"
    transform = ResizeAndPadWithBoxes((800, 800))
    dataset = CocoDetectionTransform(root=root, annFile=ann, transform=transform)
    #print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    #checkpoint = torch.load('/content/drive/MyDrive/AI/fasterrcnn_avatar_checkpoint.pth')
    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace the classifier with a new one (num_classes = background + your classes)
    num_classes = 3  # 2 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_history = []
    model.train()
    for epoch in range(50):
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Skip batches with empty targets
            targets = [t for t in targets if len(t["boxes"]) > 0]
            if len(targets) == 0:
                print(f"Skipping batch {batch_idx} - no valid targets")
                continue
            
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            loss_value = losses.item()
            loss_history.append(loss_value)
            print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss_value:.4f}")
        
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_history': loss_history,
    }, '/content/drive/MyDrive/fasterrcnn.pth')
    print("Training complete. Model saved!")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(r"/content/drive/MyDrive/loss_curve.png")  # 保存图片
    plt.show()  # 显示图片
if __name__ == "__main__":
    main()