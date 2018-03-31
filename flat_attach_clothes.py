import pickle
import cv2
import numpy as np
import time

def flat_attach_clothes(mask, original, clothes, debug=False):
    '''
    # mask: pickle file
    mask = pickle.load(open('outputs/20180331_08431522457007_mask.pickle',
                            'rb'))
    # original: original person img w. background
    original = cv2.imread('outputs/original.jpg')
    # clothes: clothes image with white background
    clothes = cv2.imread('outputs/clothes.jpg')
    '''

    # convert mask pickle to black and white jpg-like format,
    # masked part is 255 (white)
    t = time.time()
    types = [5, 6, 7]
    mask_bin = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    top, left = 99999, 99999
    bot, right = -1, -1
    row, col = mask.shape
    for i in range(0, row):
        for j in range(0, col):
            if mask[i][j] in types:
                if i < top:
                    top = i
                if i > bot:
                    bot = i
                if j < left:
                    left = j
                if j > right:
                    right = j
                mask_bin[i][j] = (255, 255, 255)
    print(time.time() - t)
    t = time.time()

    mymask = mask_bin[top:bot, left:right]
    if (debug):
        cv2.imwrite('outputs/save/mask_trim.jpg', mymask)
    print(time.time() - t)
    t = time.time()

    # new_clothes = trim_clothes(clothes)
    new_clothes = cv2.imread('outputs/save/clothes_trim.jpg')
    print(time.time() - t)
    t = time.time()
    
    # resize clothes to size of segmentation
    resized_image = cv2.resize(new_clothes, (right-left, bot-top))
    if (debug):
        cv2.imwrite('outputs/save/clothes_resize.jpg', resized_image)

    final = clothes_in_segmentation(resized_image, mymask, original, (top, left))
    print(time.time() - t)
    if (debug):
        cv2.imwrite('outputs/save/new_pic.jpg', original)
    return original


def clothes_in_segmentation(clothes, mask, original, original_start_edge, debug = False):
    # get clothes in segmentation shape
    (top, left) = original_start_edge

    cloth_rev = cv2.bitwise_not(clothes)
    filt = cv2.bitwise_and(cloth_rev, mask)
    filt = cv2.bitwise_not(filt)
    if (debug): 
        cv2.imwrite('outputs/save/tmp.jpg', filt)

    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            if sum(filt[i, j]) != 255*3:
                original[top+i, left+j] = filt[i, j]

    return original

def trim_clothes(clothes, debug = False):
    # trim white sides of target clothes
    row, col, _ = clothes.shape
    tt, ll = 99999, 99999
    bb, rr = -1, -1
    for i in range(0, row, 2):
        for j in range(0, col, 2):
            if not (clothes[i][j][0] == 255 and
                    clothes[i][j][1] == 255 and
                    clothes[i][j][2] == 255):
                if i < tt:
                    tt = i
                if i > bb:
                    bb = i
                if j < ll:
                    ll = j
                if j > rr:
                    rr = j

    # trim sleeves
    wid, hei = bb-tt, rr-ll
    new_clothes = clothes[tt+wid//20:bb-wid//20, ll+hei//6:rr-hei//6]
    if (debug):
        cv2.imwrite('outputs/save/clothes_trim.jpg', new_clothes)
    return new_clothes

if __name__ == '__main__':
    mask = pickle.load(open('outputs/20180331_08431522457007_mask.pickle', 'rb'))
    original = cv2.imread('outputs/original.jpg')
    clothes = cv2.imread('outputs/clothes.jpg')

    flat_attach_clothes(mask, original, clothes, debug = True)