import cv2
import numpy as np
from glob import glob
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count


def read_img(path):
	img = cv2.imread(path)
	img = cv2.resize(img,(64,64))
	return img

def get_from_path(path,batch_size=32):
	paths = glob(path)
	batch = np.random.choice(paths,size=batch_size)
	pool = Pool(cpu_count())
	imgs = pool.map(read_img,batch)
	pool.close()
	imgs = normalize(imgs,-1,1)
	return np.array(imgs)

def get_from_dataset(X,batch_size=32):
	nums = range(X.shape[0])
	batch_idx = np.random.choice(nums,size=batch_size)
	batch = X[batch_idx]
	# batch = normalize(batch,-1,1)
	return batch

def normalize(x,lower=0,upper=1):
	x = np.array(x,dtype=np.float32)
	x = (((upper-lower)*x-np.min(x))/(np.max(x)-np.min(x)))+lower
	return x