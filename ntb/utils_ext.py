import numpy as np
import os
import shutil
from PIL import Image
import pandas as pd
name = "MET"
lower_name = name.lower()

app = np.genfromtxt(f"/home/ludosc/common/textile_images/color_palette_met_inner.csv", delimiter=',',dtype=str)
a = set(app[1:,0])
b = set(os.listdir(f"/home/ludosc/common/textile_images/MET_textiles_padded_inner/"))

print(len(b))
print(len(a))
app = a.union(b).difference(a)
app2 = a.union(b).difference(b)
print(len(app))
print(len(app2))

df = pd.read_csv(f"/home/ludosc/common/textile_images/color_palette_met_inner.csv")
print(df.shape)
df = df[df['H1'].notnull()]
df = df[df['fname'].isin(list(b))]

print(df.shape)
df.to_csv(f"/home/ludosc/common/textile_images/color_palette_met_inner.csv", index=False)

input_dir = f"/home/ludosc/common/textile_images/MET_textiles_padded_inner"
# output_dir = f"{input_dir}_half"

# # Create output directory
# os.makedirs(output_dir) if not os.path.exists(output_dir) else None

# # Load image
# for i,filename in enumerate(sorted(os.listdir(input_dir))):
#     print(i)
#     img = Image.open(f"{input_dir}/{filename}")
#     # Resize image (half size)
#     img = img.resize((int(img.size[0]/2),int(img.size[1]/2)))
#     # Save image
#     img.save(f"{output_dir}/{filename}")


os.makedirs(f"/home/ludosc/common/textile_images/MET_textiles_padded_inner_surplus/", exist_ok=True)
# Move image in app into VA_inner_surplus
for i in app:
    shutil.move(f"/home/ludosc/common/textile_images/MET_textiles_padded_inner/"+i,f"/home/ludosc/common/textile_images/MET_textiles_padded_inner_surplus/"+i)