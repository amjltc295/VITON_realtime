# import pickle
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

    # trim white sides of target clothes
    row, col, _ = clothes.shape
    tt, ll = 99999, 99999
    bb, rr = -1, -1
    for i in range(0, row, 2):
        for j in range(0, col, 2):
            if (clothes[i][j][0] == 255 and
                    clothes[i][j][2] == 255):
                if i < tt:
                    tt = i
                if i > bb:
                    bb = i
                if j < ll:
                    ll = j
                if j > rr:
                    rr = j

    print(time.time() - t)
    t = time.time()
    # trim sleeves
    wid, hei = bb-tt, rr-ll
    new_clothes = clothes[tt+wid//20:bb-wid//20, ll+hei//6:rr-hei//6]
    if (debug):
        cv2.imwrite('outputs/save/clothes_trim.jpg', new_clothes)

    # resize clothes to size of segmentation
    resized_image = cv2.resize(new_clothes, (right-left, bot-top))
    if (debug):
        cv2.imwrite('outputs/save/clothes_resize.jpg', resized_image)

    # get clothes in segmentation shape
    cloth_rev = cv2.bitwise_not(resized_image)
    filt = cv2.bitwise_and(cloth_rev, mymask)
    filt = cv2.bitwise_not(filt)
    if (debug):
        cv2.imwrite('outputs/save/tmp.jpg', filt)
    print(time.time() - t)
    t = time.time()

    filt_w = filt.shape[0]
    filt_h = filt.shape[1]
    for i in range(filt_w):
        for j in range(filt_h):
            if sum(filt[i, j]) != 255*3:
                original[top+i, left+j] = filt[i, j]

    print(time.time() - t)
    if (debug):
        cv2.imwrite('outputs/save/new_pic.jpg', original)
    return original

# # sample code
# Now create a mask of logo and create its inverse mask also
# img2gray = cv2.cvtColor(clothes,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# print(type(ret))
# print(type(mask))
# print(mask[100])
# input()
# mask_inv = cv2.bitwise_not(mask)
# # Now black-out the area of logo in ROI
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# # Take only region of logo from logo image.
# img2_fg = cv2.bitwise_and(clothes,clothes,mask = mask)
# # Put logo in ROI and modify the main image
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
