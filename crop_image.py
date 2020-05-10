import os
import cv2
#### crop function. Crop size = size of cropped image. Shift = shift on x , y
def crop(path,cr_size=224,cr_shift=200):
    crop_size=cr_size
    crop_shift=cr_shift
    if os.path.isfile(path):
        img = cv2.imread(path)
    else:
        return False
    cnt=0
    xcount=1+img.shape[1]//crop_shift #count windows on x:
    ycount=1+img.shape[0]//crop_shift #count windows on y:
    for yi in range(ycount):
            if crop_shift*yi+crop_size<=img.shape[0]:
                y0=crop_shift*yi
                y1=crop_shift*yi+crop_size
            else:
                y0=img.shape[0]-crop_size
                y1=img.shape[0]
            for xi in range(xcount):
                    if crop_shift*xi+crop_size<=img.shape[1]:
                        x0=crop_shift*xi
                        x1=crop_shift*xi+crop_size
                    else:
                        x0=img.shape[1]-crop_size
                        x1=img.shape[1]
                    crop = img[y0:y1,x0:x1]
                    save_path=path[:-4]+'/'
                    if os.path.exists(save_path):
                        if os.path.isdir(save_path):
                                cv2.imwrite(save_path+str(cnt)+'_pic.jpg', crop)
                    else:
                        os.mkdir(save_path)
                        cv2.imwrite(save_path+str(cnt)+'_pic.jpg', crop)
                    cnt+=1
    return True
if __name__ == '__main__':
    if not crop('/home/palladdiumm/proj2020/0.jpg',224,200):
        print('file not exists!')
