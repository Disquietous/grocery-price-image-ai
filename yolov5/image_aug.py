import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),                   # horizontal flip
    iaa.Flipud(0.2),                   # vertical flip
    iaa.Affine(rotate=(-15, 15)),     # random rotation
    iaa.Multiply((0.8, 1.2)),         # brightness
    iaa.GaussianBlur(sigma=(0, 1.0))  # blur
])

# Paths
input_image_dir = "dataset/images/train"
input_label_dir = "dataset/labels/train"
output_image_dir = "dataset/images/train_aug"
output_label_dir = "dataset/labels/train_aug"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def convert_yolo_to_bbox(label_line, img_w, img_h):
    """Convert YOLO format line to pixel bounding box"""
    parts = label_line.strip().split()
    cls = int(parts[0])
    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    width = float(parts[3]) * img_w
    height = float(parts[4]) * img_h
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return cls, BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

def convert_bbox_to_yolo(cls, bbox, img_w, img_h):
    """Convert pixel bounding box back to YOLO format"""
    x_center = (bbox.x1 + bbox.x2) / 2 / img_w
    y_center = (bbox.y1 + bbox.y2) / 2 / img_h
    width = (bbox.x2 - bbox.x1) / img_w
    height = (bbox.y2 - bbox.y1) / img_h
    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

# Loop over all images
for img_file in os.listdir(input_image_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_image_dir, img_file)
    label_path = os.path.join(input_label_dir, os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    # Read image
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]

    # Read label file
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes = []
    classes = []
    for line in lines:
        cls, bbox = convert_yolo_to_bbox(line, img_w, img_h)
        classes.append(cls)
        bboxes.append(bbox)

    bbs_on_image = BoundingBoxesOnImage(bboxes, shape=image.shape)

    for i in range(3):  # generate 3 augmented copies per image
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs_on_image)
        bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

        aug_img_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
        aug_img_path = os.path.join(output_image_dir, aug_img_name)
        aug_label_path = os.path.join(output_label_dir, f"{os.path.splitext(img_file)[0]}_aug{i}.txt")

        # Save augmented image
        cv2.imwrite(aug_img_path, image_aug)

        # Write new label file
        with open(aug_label_path, "w") as f:
            for cls, bbox in zip(classes, bbs_aug.bounding_boxes):
                yolo_line = convert_bbox_to_yolo(cls, bbox, image_aug.shape[1], image_aug.shape[0])
                f.write(yolo_line)

print("✅ Augmentation complete! Check your train_aug folders.")
