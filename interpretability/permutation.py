
import numpy as np
import gc
import time

from sklearn import metrics

#PyTorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

#import attribution
from interpretability.attribution import _attribute

cuda = torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
cudnn.benchmark = True

def set_seed():
    # For reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = True
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augmentation=None, transformation=None, target_transformation=None, orientation_transformation=None):
        self.X=X
        self.Y=Y
        self.augmentation=augmentation
        self.transformation=transformation
        self.target_transformation=target_transformation
        self.orientation_transformation=orientation_transformation

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        
        x=self.X[index]
        y=self.Y[index]

        #need to transpose and manipulate the output data into: [bs x nchannel x nsample]
        #x=np.reshape(x, (x.shape[1], x.shape[2])) 
        #transpose X for pytorch, channels last
        if x.shape[1]>x.shape[0]:
            x = np.transpose(x) 

        #orientation transformation
        if self.orientation_transformation is not None:
            x = self.orientation_transformation(x)   

        #data augmentation 
        if self.augmentation is not None: 
            x=self.augmentation(x)

        #normalise, rescale etc 
        if self.transformation is not None:
            x=self.transformation(x)

        #transform y-label (to one-hot encoding etc.)
        if self.target_transformation is not None:
            y = self.target_transformation(y)    
            
        #transpose X for pytorch, channels first and in  [nchannel x nsample]
        x = np.transpose(x, (1,0)) 

        return x, y

#----------------------------------------------------------------------#
#------------------- Permutation Testing ------------------------------#
#______________________________________________________________________#
class PermutationMask(object):
    #to-do: 
    # 2. incorporate set_seed() function

    def __init__(self, mask, rstate=42, limits=[-1, 1], with_torch=False):
        self.mask=mask
        self.limits=limits
        self.with_torch=with_torch
        self.rstate=rstate
        #initalise state(s)
        if with_torch is True:
            np.random.seed(self.rstate)
        else: 
            torch.manual_seed(self.rstate)

    def __call__(self, x):
        #to-do: allow variable length sequences > 1 
        if self.mask=='zeros':
            if torch.is_tensor(x):
                mask_value=torch.zeros(len(x))
            else:
               mask_value = np.zeros(len(x))
        elif self.mask=='noise':
            if torch.is_tensor(x):
                mask_value=torch.FloatTensor(len(x)).uniform_(self.limits[0], self.limits[1])
            else:
                mask_value=np.random.uniform(self.limits[0], self.limits[1], size=len(x))
        elif self.mask=='mean':
            if torch.is_tensor(x):
                mask_value=torch.mean(x)
            else:
                mask_value=np.mean(x)     
        else: 
            if torch.is_tensor(x):
                mask_value=torch.repeat_interleave(float(self.mask), 1)
            else:
                mask_value = np.repeat(float(self.mask), 1)

        return mask_value  

def _random_permutation(X, nPerm=20, minSegLength=6):
    '''randomly mix and permute the signal'''
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1]]
        X_new[pp:pp+len(x_temp)] = x_temp
        pp += len(x_temp)
    return(X_new)

def _mask(X_batch, Ridx, i, j, nchannels,nsamples, counter, mask):
    '''mask the sample'''
    ridxs=[]
    if i<1:
        ridxs=np.unravel_index(Ridx[j, i], (nchannels,nsamples), order='C') 
    else:
        ridxs=np.unravel_index(Ridx[j, counter:i], (nchannels,nsamples), order='C')

    ridxs=(j, ) + ridxs
    #print(ridxs)

    xsample=X_batch[ridxs]
    xsample=np.array([xsample])

    X_batch[ridxs]=mask(xsample)
    
    #print('xsample:', xsample.shape)

    return X_batch, ridxs


def _evaluate(model, X_batch, Y_batch,device='cpu'):
    '''evaluate and calculate some performance metrics'''
    #to-do: return the metrics in the dictionary and concatonate
    softmax = nn.Softmax(dim=1)
    X_batch, Y_batch = X_batch.to(device, dtype=torch.float), Y_batch.to(device,  dtype=torch.long)
    logits = model(X_batch)

    acc=metrics.accuracy_score(torch.argmax(Y_batch, dim=1).view(-1).cpu().detach().numpy(), torch.argmax(softmax(logits), dim=1).view(-1).cpu().detach().numpy())
    f1=metrics.f1_score(torch.argmax(Y_batch, dim=1).view(-1).cpu().detach().numpy(), torch.argmax(softmax(logits), dim=1).view(-1).cpu().detach().numpy(), average='weighted')

    return acc, f1

def _mask_on_batch(X_batch, Y_batch, Ridx, model, mask, sample_range_pc=1, sample_step=1, device='cpu', verbose=0):
    '''mask the samples on batch, iteratively over a portion of the signal'''
    bs, nchannels,nsamples=X_batch.shape
    
    sample_range=int(np.floor(nchannels*nsamples*sample_range_pc))
    counter=0
    acc_, f1_=[], []
    for i in range(0,sample_range+1,sample_step):
        start = time.time()
        X_batch=X_batch.cpu().detach().numpy()

        for j in range(bs):
            X_batch, ridxs=_mask(X_batch, Ridx, i, j, nchannels,nsamples, counter, mask)

        X_batch=torch.from_numpy(X_batch) 
        acc, f1=_evaluate(model, X_batch, Y_batch, device)

        acc_.append(acc)
        f1_.append(f1)
        #kappa_.append([kappa])
        #auroc_.append([auroc])

        counter=i
        '''
        if verbose and (i % 2 == 0):
                print('samples: {:} ({:.2f}%) | {:.2f}% completed | time elapsed {:.2f} [s]'.format(
                                i, 100*i/sample_range, 100*(j*i)/(bs*sample_range), time.time()-start))
        '''

    return acc_, f1_

def permutation_analysis(model, generator, analysis_model, mask, random=False, sample_range_pc=0.55, sample_step=2, device='cpu', verbose=0, AoT=False):
        '''Permutation Analysis'''
        if verbose:
            print('-------------------------- starting permutation analysis --------------------------')
        
        fcn_start = time.time()

        acc, f1, kappa, auroc=[],[],[], []
        #Forward pass
        model.eval()
        with torch.no_grad():
            for v, data in enumerate(generator):
                
                batch_start = time.time()
                X_batch, Y_batch = data
                
                bs, nchannels, nsamples=X_batch.shape

                # Transfer to GPU/CPU
                X_batch.requires_grad=True
          
            
                acc_, f1_, kappa_, auroc_=[], [], [], []   
                _acc, _f1=_evaluate(model, X_batch, Y_batch, device)

                acc_.append(_acc)
                f1_.append(_f1)
       
                #---------------------------------- (option 1) random --------------------------------------------#
                
                if (random is True) or (AoT is True): 
                    Ridx=[]
                    for k in range(bs):
                        
                        np.random.seed(k)

                        #AoT span: channels -> time (fwd):
                        _ridx=np.arange(nsamples*nchannels)
                        _ridx=np.reshape(_ridx, (nchannels, nsamples))
                        _ridx=np.transpose(_ridx)
                        _ridx=_ridx.reshape(-1)

                        #AoT span: channels -> time (rev):
                        #_ridx=np.flip(_ridx)

                        if random is True:
                            #Randomly destroy the AoT:
                            _ridx=_random_permutation(_ridx).astype(int)

                        Ridx.append(_ridx)

                        #Ridx.append(np.random.permutation(np.arange(nsamples*nchannels)))

                    Ridx=np.array(Ridx)
                #_________________________________________________________________________________________________#

                #------------------------------- (option 2) sort by relevance ------------------------------------#
                else:
                    targets=torch.argmax(Y_batch, dim=1).view(-1).cpu().detach().numpy().squeeze()
                    attribution_=[]
                    for tgt in np.unique(targets):

                        parameters=analysis_model['parameters']
                        parameters['target']=int(tgt)

                        #BUG: need to reinitialise the LRP rules every iteration as the model returns to default each loop?  
                        if 'lrp_rule' in analysis_model:
                            #print('applying lrp rule')
                            lrp_rule=analysis_model['lrp_rule']
                            model, _=lrp_rule(model)
                            #_print_lrp_rules(model)                        
                        
                        attribution=_attribute(X_batch, model, analysis_model['algorithm'], **parameters)
                        
                        ''' #to-do (AC)
                        1. layer attribution (interpolation)
                        if attribution.shape[-1] != nsamples:
                            attribution=torch.stack([LayerAttribution.interpolate(attribution, (nsamples)), LayerAttribution.interpolate(attribution, (nsamples)), LayerAttribution.interpolate(attribution, (nsamples))], dim=1)
                        '''
                        attribution=attribution.squeeze().cpu().detach().numpy()
                        attribution_.append([attribution])

                    attribution_=np.array(attribution_).squeeze()
                    #print('att (shape):', attribution_.shape)
                    if attribution_.ndim > 3:
                        #print('applying target rule')
                        Ri=[attribution_[targets[index_in_batch], index_in_batch, :, :] for index_in_batch in range(bs)]
                    else:
                        #print('NOT applying target rule')
                        Ri=attribution_

                    Ri=np.array(Ri).squeeze()

                    Rsamp, Ridx=[], [] 
                    for s in range(bs):
                        ri=Ri[s, :, :].reshape(-1, order='C')
                        #idxs=np.argsort(ri)
                        idxs=np.argsort(-ri)
                        #idxs=np.argsort(-abs(ri))
                        Rsamp.append(ri)
                        Ridx.append(idxs)
                    Rsamp=np.array(Rsamp)
                    Ridx=np.array(Ridx)
                #_________________________________________________________________________________________________#
    
                
                acc_, f1_=_mask_on_batch(X_batch, Y_batch, Ridx, model, mask, sample_range_pc=sample_range_pc, sample_step=sample_step, device=device, verbose=verbose)

                #To-Do: dump these values in future release
                acc.append(acc_)
                f1.append(f1_)
                #kappa.append([kappa_)
                #auroc.append(auroc_)

                if verbose and (v % 2 == 0):
                    print('batch: {:}/{:} | time elapsed: {:.2f} [s]'.format(v+1, len(generator), time.time()-batch_start))

        #make into a numpy
        acc=np.array(acc).squeeze()
        kappa=np.array(kappa).squeeze()
        f1=np.array(f1).squeeze()
        auroc=np.array(auroc).squeeze()
        sample_range=nsamples*nchannels
        samples=100*np.arange(0, sample_range*sample_range_pc+1, sample_step)/sample_range
        
        if verbose:
            print('------------------------- completed  permutation analysis -------------------------\ntotal time elapsed {:3.2f} [sec] | {:3.2f} [min]'.format(time.time()-fcn_start, (time.time()-fcn_start)/60))

        return {'samples': samples, 'acc': acc, 'kappa': kappa, 'f1': f1, 'auroc': auroc}

def pchange(x, baseline):
    #if need to measure this as a % (-) change from baseline
    return ((x-baseline)/np.absolute(baseline))*100