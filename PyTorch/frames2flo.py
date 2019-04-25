import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import models
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""
def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())
        
def operate(im1_fn,im2_fn,flow_fn):
  if len(sys.argv) > 1:
      im1_fn = sys.argv[1]
  if len(sys.argv) > 2:
      im2_fn = sys.argv[2]
  if len(sys.argv) > 3:
      flow_fn = sys.argv[3]

  pwc_model_fn = './pwc_net.pth.tar';

  im_all = [imread(img) for img in [im1_fn, im2_fn]]
  im_all = [im[:, :, :3] for im in im_all]

  # rescale the image size to be multiples of 64
  divisor = 64.
  H = im_all[0].shape[0]
  W = im_all[0].shape[1]

  H_ = int(ceil(H/divisor) * divisor)
  W_ = int(ceil(W/divisor) * divisor)
  for i in range(len(im_all)):
    im_all[i] = cv2.resize(im_all[i], (W_, H_))

  for _i, _inputs in enumerate(im_all):
    im_all[_i] = im_all[_i][:, :, ::-1]
    im_all[_i] = 1.0 * im_all[_i]/255.0

    im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
    im_all[_i] = torch.from_numpy(im_all[_i])
    im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
    im_all[_i] = im_all[_i].float()

  im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

  net = models.pwc_dc_net(pwc_model_fn)
  net = net.cuda()
  net.eval()

  flo = net(im_all)
  flo = flo[0] * 20.0
  flo = flo.cpu().data.numpy()

  # scale the flow back to the input size 
  flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
  u_ = cv2.resize(flo[:,:,0],(W,H))
  v_ = cv2.resize(flo[:,:,1],(W,H))
  u_ *= W/ float(W_)
  v_ *= H/ float(H_)
  flo = np.dstack((u_,v_))

  writeFlowFile(flow_fn, flo)


import os
frames_dir = "./Frames_data/00007/"
save_dir = "./flo_data/00007/"
files_set = []
frames_set = []
IDs = []
for filename in os.listdir(frames_dir):
  files_set.append(filename)
files_set.sort()
  
for i in range(len(files_set)-1):
  if files_set[i][0:10] == files_set[i+1][0:10]:
    frames_set.append([files_set[i],files_set[i+1]])
    IDs.append(files_set[i])
  else:
    continue
  
# im1_fn = 'data/frame_0010.png';
# im2_fn = 'data/frame_0011.png';
# flow_fn = './tmp/frame_0010.flo';
for i in range(len(frames_set)):
  im1_fn = frames_dir + frames_set[i][0]
  im2_fn = frames_dir + frames_set[i][1]
  flow_fn = save_dir + IDs[i][0:-4] + ".flo"
  operate(im1_fn,im2_fn,flow_fn)
