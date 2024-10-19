import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import argparse
import os

import torch
from tqdm import tqdm
import sys


img = plt.imread("/NBA-Player-Movements/court.png")



shooter = {1: 'b', 2: 'b', 3: 'b', 4: 'b', 5: 'b'}

if (len(sys.argv) == 6):
    file = sys.argv[1].split('/')[1]
    sample = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    shooter[int(sys.argv[5])] = 'r'
elif (len(sys.argv) == 5):
    file = sys.argv[1].split('/')[1]
    sample = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])
else:
    sys.exit()
    
path_trajectory = f'/diffuser_bball/logs/{sys.argv[1].split("/")[0]}'
print(f'{path_trajectory}/{file}.npy')
with open(f"{path_trajectory}/{file}.npy", 'rb') as fin:
    observations = torch.load(fin)

alphas = [1, 1, 1]      # this is only for better reading, 0.1 means past, 1 means nearer

# Every trajectory contains 5 samples, we visualize all and only need to find 1 ideal case

plt.figure(figsize=(8,5))
plt.xlim(0, 94)
plt.ylim(0, 50)
plt.imshow(img, extent=[0, 94, 0, 50])
plt.xticks([])
plt.yticks([])
for i in range(start, end):
    if i < 300:
        alpha = alphas[0]
    elif i >= 300 and i < 600:
        alpha = alphas[1]
    else:
        alpha = alphas[2]
    plt.scatter(observations[sample][i][0], observations[sample][i][1], color='#ee6730', alpha=alpha, s=5)
    plt.scatter(observations[sample][i][6], observations[sample][i][7], color=shooter[1], alpha=alpha, s=5)
    plt.scatter(observations[sample][i][12], observations[sample][i][13], color=shooter[2], alpha=alpha, s=5)
    plt.scatter(observations[sample][i][18], observations[sample][i][19], color=shooter[3], alpha=alpha, s=5)
    plt.scatter(observations[sample][i][24], observations[sample][i][25], color=shooter[4], alpha=alpha, s=5)
    plt.scatter(observations[sample][i][30], observations[sample][i][31], color=shooter[5], alpha=alpha, s=5)

for i in range(start, end):
    if i < 300:
        alpha = alphas[0]
    elif i >= 300 and i < 600:
        alpha = alphas[1]
    else:
        alpha = alphas[2]
    plt.scatter(observations[sample][i][36], observations[sample][i][37], color='black', alpha=alpha, s=5)
    plt.scatter(observations[sample][i][42], observations[sample][i][43], color='black', alpha=alpha, s=5)
    plt.scatter(observations[sample][i][48], observations[sample][i][49], color='black', alpha=alpha, s=5)
    plt.scatter(observations[sample][i][54], observations[sample][i][55], color='black', alpha=alpha, s=5)
    plt.scatter(observations[sample][i][60], observations[sample][i][61], color='black', alpha=alpha, s=5)



path_parts = path_trajectory.split('/logs/')
subfolder = path_parts[1]

save_folder = os.path.join('/NBA-Player-Movements/shooter_png_dir/good_png/', subfolder)

# Create the target directory if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

plt.tight_layout()
print('saving...')
plt.savefig(f'/NBA-Player-Movements/shooter_png_dir/good_png/{subfolder}/{file}_{sample}.png')
plt.close()
