import numpy as np
from PIL import Image

list_im = ['a_0.jpg', 's.jpg', 'd.jpg', 'f.jpg']
imgs = [Image.open(i) for i in list_im]
# pick the image which is the smallest,
# and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

# save that beautiful picture
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save('Trifecta.jpg')

# for a vertical stacking it is simple: use vstack
imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
imgs_comb = Image.fromarray(imgs_comb)
imgs_comb.save('Trifecta_vertical.jpg')
