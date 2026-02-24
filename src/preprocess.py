from torchvision import transforms


def get_preprocess(checkpoint):

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    img_size = checkpoint["img_size"]

    def crop_vertical_15_percent(img):
        w, h = img.size
        top = int(0.15 * h)
        bottom = int(0.85 * h)
        return img.crop((0, top, w, bottom))

    return transforms.Compose([
        transforms.Lambda(crop_vertical_15_percent),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])