#!/usr/bin/env python
# coding: utf-8


# This cell imports libraries that you will need
# Run this.
from matplotlib.pyplot import figure, cm
import os
import numpy as np
import utils1
import h5py
from sklearn.model_selection import KFold
import logging
import argparse
import pandas as pd
import numexpr as ne
logging.basicConfig(level=logging.DEBUG)

class TRFile(object):
    def __init__(self, trfilename, expectedtr=1.5):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        
        if trfilename is not None:
            self.load_from_file(trfilename)
        

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        ## Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label=="sound-start":
                self.soundstarttime = time

            elif label=="sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))
        
        ## Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes>(itrtimes.mean()*1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            ## Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime,btr))

        for ntr,btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)
    
    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def load_generic_trfiles(stories, root="./features_21styear/"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.
    """
    trdict = dict()

    for story in stories:
        try:
            trf = TRFile(os.path.join(root, "%s.report"%story))
            trdict[story] = [trf]
        except Exception as e:
            print (e)
    
    return trdict


trfiles = load_generic_trfiles(['21styear'])

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.
    
    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.
    
    [window] lobes of the sinc function will be used. [window] should be an integer.
    """
    ## Find the cutoff frequency ##
    cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
    print ("Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))
    
    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
    
    if rectify:
        newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
                            np.dot(sincmat, np.clip(data, 0, np.inf))])
    else:
        ## Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata

def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.
    """
    t = t * cutoff
    pi = np.pi
    #val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
    val = ne.evaluate("window * sin(pi*t) * sin(pi*t/window) / (pi**2 * t**2)")
    val[t==0] = 1.0
    val[np.abs(t)>window] = 0.0

    return val

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
	parser.add_argument("subjectNum", help="Choose subject", type = int)
	parser.add_argument("layers", help="Choose layers", type = int)
	parser.add_argument("featurename", help="Choose feature", type = str)
	parser.add_argument("context", help="Choose context", type = int)
	parser.add_argument("outputdir", help="Choose layers", type = str)
	args = parser.parse_args()

	num_layers = args.layers

	#load feature name
	word_level_features = np.load('./features_21styear/'+args.featurename, allow_pickle=True)

	#load tunnel alignment
	word_alignment = pd.read_csv('./features_21styear/21styear_align.csv', header=None)
	word_alignment.head()


	word_alignment[[2, 3]] = word_alignment[[2, 3]].apply(pd.to_numeric)
	word_alignment['word_times'] = (word_alignment[2]+word_alignment[3])/2

	TR_aligned_features = []
	for eachlayer in np.arange(num_layers):
	    aligned_features = lanczosinterp2D(word_level_features.item()[eachlayer], word_alignment['word_times'],trfiles['21styear'][0].trtimes, window=3)
	    TR_aligned_features.append(aligned_features)

	from npp import zscore
	Rstim = []
	for eachlayer in np.arange(num_layers):
	    Rstim.append(np.vstack([np.array(TR_aligned_features[eachlayer][14:-9])]))

	# Delay stimuli
	from util import make_delayed
	ndelays = 8
	delays = range(1, ndelays+1)

	print ("FIR model delays: ", delays)

	delRstim = []
	for eachlayer in np.arange(num_layers):
	    delRstim.append(make_delayed(Rstim[eachlayer], delays))


	# In[50]:


	# Print the sizes of these matrices
	print ("delRstim shape: ", delRstim[0].shape)


	# Load training data for subject 1, reading dataset 
	roi_voxels = np.load('./21styear/sub_'+str(args.subjectNum)+'.npy',allow_pickle=True)
	roi_voxels = roi_voxels[14:-9,:]
	print(roi_voxels.shape)


	# In[26]:


	from npp import zscore
	zRresp = []
	for eachsubj in np.arange(roi_voxels.shape[0]):
	    zRresp.append(roi_voxels[eachsubj])
	zRresp = np.array(zRresp)


	# Run regression
	from ridge_utils.ridge import bootstrap_ridge
	from scipy import stats
	from sklearn.model_selection import KFold
	#alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
	nboots = 1 # Number of cross-validation runs.
	chunklen = 40 # 
	nchunks = 20
	kf = KFold(n_splits=4)
	save_dir = args.outputdir
	if not os.path.exists(save_dir):
	    os.mkdir(save_dir)

	subdir = str(args.context)
	if not os.path.exists(save_dir+'/'+subdir):
	    os.mkdir(save_dir+'/'+subdir)

	#for eachlayer in np.arange(len(word_level_features.item())):
	for eachlayer in np.arange(num_layers):
	    if not os.path.exists(save_dir+'/'+subdir+'/'+str(eachlayer)):
        	os.mkdir(save_dir+'/'+subdir+'/'+str(eachlayer))
	    for eachsub in np.arange(args.subjectNum,args.subjectNum+1):
	        count = 0
	        all_preds = []
	        all_reals = []
	        all_corrs = []
	        all_acc = []
	        for train_index, test_index in kf.split(zRresp):
	                alphas = np.logspace(1, 3, 10)
	                # remove 5 TRs which either precede or follow the TRs in the test set

	                train_index_without_overlap = train_index
	                for rem_val in range(test_index[0] - 5, test_index[0], 1):
	                    train_index_without_overlap = train_index_without_overlap[train_index_without_overlap != rem_val]

	                for rem_val in range(test_index[-1] + 1, test_index[-1] + 6, 1):
	                    train_index_without_overlap = train_index_without_overlap[train_index_without_overlap != rem_val]

	                x_train, x_test = delRstim[eachlayer][train_index_without_overlap], delRstim[eachlayer][test_index]
	                y_train, y_test = zRresp[train_index_without_overlap], zRresp[test_index]
	                
	                x_train = zscore(x_train)
	                x_test = zscore(x_test)
	                y_train = zscore(y_train)
	                y_test = zscore(y_test)
	                print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	                wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(x_train, y_train, x_test, y_test,
	                                                                     alphas, nboots, chunklen, nchunks,
	                                                                     singcutoff=1e-10, single_alpha=True)
	                y_pred = np.dot(x_test, wt)
	                print ("pred has shape: ", y_pred.shape)
	                #np.save(os.path.join(save_dir+'/'+subdir+'/'+str(eachlayer), "y_pred_{}".format(count)),y_pred)
	                #np.save(os.path.join(save_dir+'/'+subdir+'/'+str(eachlayer), "y_test_{}".format(count)),y_test)
	                all_reals.append(y_test)
	                all_preds.append(y_pred)
	                all_corrs.append(corr)
	                #all_acc.append(binary_classify_neighborhoods(y_pred, y_test))

	                count+=1

	        all_reals = np.vstack(all_reals)
	        all_preds = np.vstack(all_preds)
	        all_corr = np.vstack(all_corrs)

	        voxcorrs = np.zeros((all_reals.shape[1],)) # create zero-filled array to hold correlations
	        for vi in range(all_reals.shape[1]):
	            voxcorrs[vi] = np.corrcoef(all_reals[:,vi], all_preds[:,vi])[0,1]
	        print (voxcorrs)
	        print(np.mean(voxcorrs[np.where(voxcorrs>0)[0]]))

	        np.save(os.path.join(save_dir+'/'+subdir+'/'+str(eachlayer), "subj_"+str(eachsub)+"_layer_"+str(eachlayer)),np.mean(all_corr,axis=0))
	        np.save(os.path.join(save_dir+'/'+subdir+'/'+str(eachlayer), "subj1_"+str(eachsub)+"_layer_"+str(eachlayer)),voxcorrs)