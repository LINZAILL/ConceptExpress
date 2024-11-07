
from PIL import Image, ImageOps, ImageChops
from torchvision import transforms
import numpy as np

size = 512
image_transforms = transforms.Compose(
[
    transforms.Resize([size,size]),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
]
)
upsampling = transforms.Resize([512,512])

#translate (shift)
def ImgOfffSet(Img,xoff,yoff):
    width, height = Img.size
    c = ImageChops.offset(Img,xoff,yoff)
    c.paste((0,0,0),(0,0,xoff,height))
    c.paste((0,0,0),(0,0,width,yoff))
    return c

def MaskOfffSet(Img,xoff,yoff):
    width, height = Img.size
    c = ImageChops.offset(Img,xoff,yoff)
    # c.paste((0),(0,0,xoff,height))
    # c.paste((0),(0,0,width,yoff))
    return c

instance_img_path_1 = "sam_final_masked0_78.png"
instance_img_path_2 = "sam_final_masked0_88.png"
image_1 = Image.open(instance_img_path_1)
image_2 = Image.open(instance_img_path_2)
# image_1 = image_transforms(image_1)
# image_2 = image_transforms(image_2)

image_1 = upsampling(image_1)
image_2 = upsampling(image_2)

image_1_shifted = ImgOfffSet(image_1, 0, 20 * 2)
image_2_shifted = ImgOfffSet(image_2, 110 * 2, 0)

image_1_shifted_np = np.array(image_1_shifted)
image_2_shifted_np = np.array(image_2_shifted)
image_new_np = image_1_shifted_np + image_2_shifted_np
# image_new = transforms.ToPILImage()(image_new_np)
image_new = Image.fromarray(image_new_np)

# image_new.save("temp.png")
image_new.save("img_combined.jpg")

instance_mask_path_1 = "sam_mask_all0_78.png"
instance_mask_path_2 = "sam_mask_all0_88.png"
mask_1 = Image.open(instance_mask_path_1)
mask_2 = Image.open(instance_mask_path_2)
mask_1 = upsampling(mask_1)
mask_2 = upsampling(mask_2)

mask_1_shifted = MaskOfffSet(mask_1, 0, 20 * 2)
mask_2_shifted = MaskOfffSet(mask_2, 110 * 2, 0)
mask_1_shifted_np = np.array(mask_1_shifted)
mask_2_shifted_np = np.array(mask_2_shifted)
mask_new_np = mask_1_shifted_np + mask_2_shifted_np
mask_new = Image.fromarray(mask_new_np)

mask_new.save("mask_combined.jpg")

