from PIL import Image
import glob
image_list = []
for filename in glob.glob('RedChair/*.jpg'):
    im = Image.open(filename).convert('LA')
    image_list.append(im)
    
print image_list