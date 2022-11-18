from shutil import copyfile
import os.path

cnt = 0

for i in range (2685, 3286):

    original_label_file = f'image_processing/train/labels/{i}.txt'

    if os.path.isfile(original_label_file):
        new_label_file = f'image_processing/valid/labels/{cnt}.txt'
        copyfile(original_label_file, new_label_file)
        
    cnt += 1

print("Copy complete")