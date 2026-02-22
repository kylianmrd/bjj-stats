import os
from PIL import Image
from torchvision import transforms

# Transformations d'augmentation
augment = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])

dataset_path = "dataset"

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        
        for i in range(5):  # 5 nouvelles images par image
            aug_img = augment(img)
            new_name = f"aug_{i}_{img_name}"
            aug_img.save(os.path.join(class_path, new_name))

print("Augmentation terminée.")
