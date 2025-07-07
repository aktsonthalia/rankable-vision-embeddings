from pycocotools.coco import COCO

# in img_path files, let each image appear as many times as the number of categories of objects in the image
# 
# Load annotation file
ann_file = '/mnt/lustre/work/bethge/asonthalia61/rankanything/data/coco_rem/instances_valrem.json'
coco = COCO(ann_file)

# Get category names
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print("Categories:", cat_names)

# Get all image IDs
img_ids = coco.getImgIds()

# Pick one image
img_id = img_ids[1000]
coco_url = coco.imgs[img_id]['coco_url']
print("Image URL:", coco_url)
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# Count objects per category in this image
from collections import Counter
counts = Counter([ann['category_id'] for ann in anns])
category_id_to_name = {cat['id']: cat['name'] for cat in cats}

# Print results
for cat_id, count in counts.items():
    print(f"{category_id_to_name[cat_id]}: {count}")

# show img url