import numpy as np
from torch.utils.data.dataset import Subset

def randomlyMakeTrainValSets_valNet(_aug_trainset, _valset, _val_set_size ): #_trainset_rate):
    _all_shuffled_indices = np.arange(len(_aug_trainset))
    np.random.shuffle( _all_shuffled_indices )  # shuffle the original_trainset indices
    _len_trainset = int( len(_aug_trainset) - _val_set_size ) # *  _trainset_rate )
    _trainset_indices = _all_shuffled_indices[0 : _len_trainset]
    _valset_indices = _all_shuffled_indices[_len_trainset : len(_aug_trainset)]

    _aug_trainset = Subset(_aug_trainset, _trainset_indices)
    _valset = Subset(_valset, _valset_indices)
    return _aug_trainset, _valset
