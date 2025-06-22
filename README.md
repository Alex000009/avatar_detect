avatar-detect/
├── images/              # 示例图像文件夹（部分展示）
│   ├── train
│   └── ...
├── annotations/          # COCO 格式标注文件（train.json, val.json）
├── train.py              # 模型训练脚本
├── val.py                # 验证评估脚本
├── test.py               # 单图测试并可视化
├── xml_to_coco.py        # VOC 转 COCO 格式转换工具
├── fasterrcnn_16.pth             # 已训练模型（使用 Git LFS 管理）
├── requirements_list.txt # 项目依赖库
├── README.md

环境依赖在requirements_list.txt中大部分都有，能力有限，如有问题或错误请联系1761694505@qq.com
运行时需要将代码里面的文件路径修改好，比如train.py 中root = r"D:\Desktop\AI\project\images\train" 要改为你下载好的路径。
