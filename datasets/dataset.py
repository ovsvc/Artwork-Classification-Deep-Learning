from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Tuple



class Subset(Enum):
    '''
    Dataset subsets.
    '''

    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class Dataset(metaclass=ABCMeta):
    '''
    Base class of all datasets.
    '''

    @abstractmethod
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        pass

