import cv2
import numpy as np
from img_aug import Img_aug #데이터 증강 class를 불러옴

# augmentation_path = 'image_processing/image_augmentation/aug_file/'

aug = Img_aug()		#데이터 증강 class 선언
augment_num = 2	    #증강결과로 출력되는 이미지의 갯수 선언
save_path = 'image_processing/train'

whole_num_of_image = 1095

for i in range(whole_num_of_image):
    jpg_path = f'image_processing/train/images/{i}.jpg'
    img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    images_aug = aug.seq.augment_images([img for i in range(augment_num)])
    
    for num,aug_img in enumerate(images_aug) :
        cv2.imwrite(save_path+'/augmented_images/'+f'{aug.cnt}.jpg',aug_img)
        aug.cnt += 1

print('Complete augmenting images')