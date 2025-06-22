import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def visualize_predictions(image_tensor, boxes, labels, scores, label_map, threshold=0.5):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f'{label_map.get(label, label)}: {score:.2f}', color='white', 
                fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"pred_{image_id}.png")  # Optional: save to file


# Custom dataset class
class CocoTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        return F.to_tensor(image), img_id

    def __len__(self):
        return len(self.ids)

def main():
    # Paths
    test_img_dir = r"D:\Desktop\AI\project\images\test"
    ann_file = r"D:\Desktop\AI\project\annotations\instances_test.json"
    model_path = r"D:\Desktop\AI\fasterrcnn_8.pth"
    result_json = "results_test.json"
    label_map = {1: 'Korra', 2: 'Mako'}
    korra_id = 1
    Mako_id = 2
    # Load dataset
    dataset = CocoTestDataset(test_img_dir, ann_file)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)  # 2 classes + background
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Run inference
    coco_results = []
    with torch.no_grad():
        for images, image_ids in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for image, output, img_id in zip(images, outputs, image_ids):
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                visualize_predictions(image.detach(), boxes, labels,scores, label_map)
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_results.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [float(x_min),float(y_min), float(width), float(height)],
                    "score": float(score)
                    })
                
    # Save results
    with open(result_json, "w") as f:
        json.dump(coco_results, f, indent=4)

    # Evaluate with COCO API
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(result_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
   

if __name__ == "__main__":
    main()
