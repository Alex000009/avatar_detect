import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
from torchvision import transforms

# --------- Custom Dataset Class ---------
class CocoDetectionTransform(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self._transform = transform

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for obj in anns:
            bbox = obj['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self._transform is not None:
            img = self._transform(img)

        return img, target

# --------- Utility Functions ---------
def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    return transforms.Compose([transforms.ToTensor()])

def draw_boxes(image_np, boxes, labels, scores, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image_np)
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f'{label}: {score:.2f}', color='yellow', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.show()

# --------- Main Validation ---------
def main():
    val_root = r"D:\Desktop\AI\project\images\val"
    val_ann = r"D:\Desktop\AI\project\annotations\instances_val.json"
    model_path = r"D:\Desktop\AI\fasterrcnn_16.pth"

    dataset = CocoDetectionTransform(val_root, val_ann, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weithts=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    coco_gt = dataset.coco
    coco_results = []

    print("Evaluating...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = int(targets[i]["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Convert to COCO result format
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    coco_results.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(score)
                    })

                # --- Visualization (optional) ---
                img_path = os.path.join(val_root, coco_gt.loadImgs(image_id)[0]['file_name'])
                image_np = np.array(Image.open(img_path).convert("RGB"))
                draw_boxes(image_np, boxes, labels, scores, threshold=0.5)

    # Save results to file for COCOeval
    result_file = "coco_val_results.json"
    with open(result_file, "w") as f:
        json.dump(coco_results, f, indent=4)

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()

