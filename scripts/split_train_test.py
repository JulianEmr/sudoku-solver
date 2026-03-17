import os, shutil, random

random.seed(42)

images = os.listdir("data/label-studio/images")
random.shuffle(images)

split = int(0.85 * len(images))
train_imgs = images[:split]
val_imgs = images[split:]

# build a lookup dict: "sudoku_036" -> "09292718-sudoku_036.txt"
label_lookup = {}
for f in os.listdir("data/label-studio/labels"):
    if f.startswith('.'):
        continue
    clean_name = f.split("-", 1)[1].rsplit(".", 1)[0]  # "09292718-sudoku_036.txt" → "sudoku_036"
    label_lookup[clean_name] = f

for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
    os.makedirs(f"data/{split_name}/images", exist_ok=True)
    os.makedirs(f"data/{split_name}/labels", exist_ok=True)
    
    for img_file in split_imgs:
        # copy image
        shutil.copy(f"data/label-studio/images/{img_file}", f"data/{split_name}/images/{img_file}")
        
        # find matching label via lookup
        img_stem = img_file.rsplit(".", 1)[0]  # "sudoku_036.png" → "sudoku_036"
        if img_stem in label_lookup:
            src = f"data/label-studio/labels/{label_lookup[img_stem]}"
            dst = f"data/{split_name}/labels/{img_stem}.txt"  # save without the hash
            shutil.copy(src, dst)
        else:
            print(f"⚠️  No label found for {img_file}")

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")