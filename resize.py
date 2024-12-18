# 提取目录下所有图片,更改尺寸后保存到另一目录

from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=480,height=320):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
# for jpgfile in glob.glob("outputs/data/train_VGG_UNet_woDS/0\\*.jpg"):
#     convertjpg(jpgfile,"./outputs/data/train_VGG_UNet_woDS/0")
for jpgfile in glob.glob("outputs/data/train_Resnet50_UNet_woDS/0\\*.jpg"):
    convertjpg(jpgfile, "./outputs/data/train_Resnet50_UNet_woDS/0")