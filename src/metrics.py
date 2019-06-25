import numpy as np

import sys
import numpy as np
import scipy.spatial.distance as ssd
#from tstab import *

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
	"""
	Compute mean,recall and decay from per-frame evaluation.
	Calculates precision/recall for boundaries between foreground_mask and
	gt_mask using morphological operators to speed it up.

	Arguments:
		foreground_mask (ndarray): binary segmentation image.
		gt_mask         (ndarray): binary annotated image.

	Returns:
		F (float): boundaries F-measure
		P (float): boundaries precision
		R (float): boundaries recall
	"""
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask);
	gt_boundary = seg2bmap(gt_mask);

	from skimage.morphology import binary_dilation,disk

	fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg     = np.sum(fg_boundary)
	n_gt     = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall    = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall);

	return F

def seg2bmap(seg,width=None,height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.

	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]

	Returns:
		bmap (ndarray):	Binary boundary map.

	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+floor((y-1)+height / h)
					i = 1+floor((x-1)+width  / h)
					bmap[j,i] = 1;

	return bmap


def get_bijective_pairs(pairs,costmat):
	bij_pairs = bij_pairs_one_dim(pairs, costmat,0)
	bij_pairs = bij_pairs_one_dim(bij_pairs, costmat.T,1)
	return bij_pairs

def bij_pairs_one_dim(pairs, costmat, left_or_right):

	bij_pairs = []
	ids1      = np.unique(pairs[:,left_or_right])

	for ii in range(len(ids1)):
		curr_pairs = pairs[pairs[:,left_or_right]==ids1[ii],:].astype(np.int)
		curr_costs = costmat[curr_pairs[:,left_or_right], curr_pairs[:,1-left_or_right]]
		b = np.argmin(curr_costs)
		bij_pairs.append(curr_pairs[b])

	return np.array(bij_pairs)

def hist_cost_2(BH1,BH2):

	nsamp1,nbins=BH1.shape
	nsamp2,nbins=BH2.shape

	eps  = 2.2204e-16
	BH1n = BH1 / (np.sum(BH1,axis=1,keepdims=True)+eps)
	BH2n = BH2 / (np.sum(BH2,axis=1,keepdims=True)+eps)

	tmp1 = np.tile(np.transpose(np.atleast_3d(BH1n),[0,2,1]),(1,nsamp2,1))
	tmp2 = np.tile(np.transpose(np.atleast_3d(BH2n.T),[2,1,0]),(nsamp1,1,1))
	HC = 0.5*np.sum((tmp1-tmp2)**2/(tmp1+tmp2+eps),axis=2)

	return HC

def sc_compute(Bsamp,Tsamp,mean_dist,nbins_theta,nbins_r,r_inner,r_outer,out_vec):
	in_vec = (out_vec==0).ravel()
	nsamp = Bsamp.shape[1]
	r_array=ssd.squareform(ssd.pdist(Bsamp.T)).T

	theta_array_abs0=Bsamp[1,:].reshape(-1,1).dot(np.ones((1,nsamp))) - \
			np.ones((nsamp,1)).dot(Bsamp[1,:].reshape(1,-1))

	theta_array_abs1=Bsamp[0,:].reshape(-1,1).dot(np.ones((1,nsamp))) - \
			np.ones((nsamp,1)).dot(Bsamp[0,:].reshape(1,-1))

	theta_array_abs = np.arctan2(theta_array_abs0,theta_array_abs1).T
	theta_array=theta_array_abs-Tsamp.T.dot(np.ones((1,nsamp)))

	if mean_dist is None:
		mean_dist = np.mean(r_array[in_vec].T[in_vec].T)

	r_array_n = r_array / mean_dist

	r_bin_edges=np.logspace(np.log10(r_inner),np.log10(r_outer),nbins_r)
	r_array_q=np.zeros((nsamp,nsamp))

	for m in range(int(nbins_r)):
		r_array_q=r_array_q+(r_array_n<r_bin_edges[m])

	fz = r_array_q > 0
	theta_array_2 = np.fmod(np.fmod(theta_array,2*np.pi)+2*np.pi,2*np.pi)
	theta_array_q = 1+np.floor(theta_array_2/(2*np.pi/nbins_theta))

	nbins=nbins_theta*nbins_r
	BH=np.zeros((nsamp,nbins))
	count = 0
	for n in range(nsamp):
		fzn=fz[n]&in_vec
		Sn = np.zeros((nbins_theta,nbins_r))
		coords = np.hstack((theta_array_q[n,fzn].reshape(-1,1),
			r_array_q[n,fzn].astype(np.int).reshape(-1,1)))

		# SLOW...
		#for i,j in coords:
			#Sn[i-1,j-1] += 1

		# FASTER
		ids = np.ravel_multi_index((coords.T-1).astype(np.int),Sn.shape)
		Sn  = np.bincount(ids.ravel(),minlength = np.prod(Sn.shape)).reshape(Sn.shape)


		BH[n,:] = Sn.T[:].ravel()

	return BH.astype(np.int),mean_dist

def db_eval_t_stab(fgmask,ground_truth,timing=True):
	"""

	Calculates the temporal stability index between two masks

	Arguments:
					fgmask (ndarray):  Foreground Object mask at frame t
		ground_truth (ndarray):  Foreground Object mask at frame t+1

	Return:
							 T (ndarray):  Temporal (in-)stability
	   raw_results (ndarray):  Supplemental values
	"""

	cont_th = 3
	cont_th_up = 3

	# Shape context parameters
	r_inner     = 1.0/8.0
	r_outer     = 2.0
	nbins_r     = 5.0
	nbins_theta = 12.0

	poly1 = mask2poly(fgmask,cont_th)
	poly2 = mask2poly(ground_truth,cont_th)

	if len(poly1.contour_coords) == 0 or \
			len(poly2.contour_coords) == 0:
		return np.nan

	Cs1 = get_longest_cont(poly1.contour_coords)
	Cs2 = get_longest_cont(poly2.contour_coords)

	upCs1 = contour_upsample(Cs1,cont_th_up)
	upCs2 = contour_upsample(Cs2,cont_th_up)

	scs1,_=sc_compute(upCs1.T,np.zeros((1,upCs1.shape[0])),None,
			nbins_theta,nbins_r,r_inner,r_outer,np.zeros((1,upCs1.shape[0])))

	scs2,_=sc_compute(upCs2.T,np.zeros((1,upCs2.shape[0])),None,
			nbins_theta,nbins_r,r_inner,r_outer,np.zeros((1,upCs2.shape[0])))

	# Match with the 0-0 alignment
	costmat              = hist_cost_2(scs1,scs2)
	pairs ,max_sx,max_sy = match_dijkstra(np.ascontiguousarray(costmat))


	# Shift costmat
	costmat2 = np.roll(costmat ,-(max_sy+1),axis=1)
	costmat2 = np.roll(costmat2,-(max_sx+1),axis=0)

	# Redo again with the correct alignment
	pairs,_,_ = match_dijkstra(costmat2)

	# Put the pairs back to the original place
	pairs[:,0] = np.mod(pairs[:,0]+max_sx+1, costmat.shape[0])
	pairs[:,1] = np.mod(pairs[:,1]+max_sy+1, costmat.shape[1])

	pairs = get_bijective_pairs(pairs,costmat)

	pairs_cost = costmat[pairs[:,0], pairs[:,1]]
	min_cost   = np.average(pairs_cost)

	return min_cost


def db_eval_iou(annotation,segmentation):

	""" Compute region similarity as the Jaccard Index.

	Arguments:
		annotation   (ndarray): binary annotation   map.
		segmentation (ndarray): binary segmentation map.

	Return:
		jaccard (float): region similarity

 """

	annotation   = annotation.astype(np.bool)
	segmentation = segmentation.astype(np.bool)

	if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
		return 1
	else:
		return np.sum((annotation & segmentation)) / \
				np.sum((annotation | segmentation),dtype=np.float32)

    
