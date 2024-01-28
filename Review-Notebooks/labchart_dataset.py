import os
import errno
import numpy as np
from scipy.io import loadmat

class LabChartDataset:
    def __init__(self, mat_path: str) -> None:
        '''
        DataSet object which acts as a container for LabChart channel data
        that has been exported as a MATLAB file.
        
        Example usage:
            DataSet = DataSet(file_path)\n
            channel_data = DataSet.data['Channel #']
        '''

        # scipy throw an error w/o this, but this should be less verbose of an error
        if os.path.exists(mat_path) == False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mat_path)
        
        self.mat_dict = loadmat(file_name=mat_path)
        self.n_channels = self.mat_dict['titles'].shape[0]
        
        self.data = {
            f'Channel {ch + 1}' : self._split_blocks(ch) for ch in range(self.n_channels)
        }

    def _split_blocks(self, channel_idx: int) -> list[np.ndarray]:
        '''
        Private method that is used to build the self.data dictionary
        '''
        
        raw = self.mat_dict['data'].reshape(-1)
        channel_starts = self.mat_dict['datastart'][channel_idx] - 1
        channel_ends = self.mat_dict['dataend'][channel_idx]
        
        n_blocks = channel_starts.shape[0]
        channel_blocks = []
        for idx in range(n_blocks):
            start = int(channel_starts[idx])
            end = int(channel_ends[idx])
            channel_blocks.append(raw[start : end])

        return channel_blocks
    
    def get_block(self, block_index: int) -> np.ndarray:
        '''
        Given a block index number, returns a (channel x timepoints) array
        containing the data for that block. If only 1 channel, returns
        a 1D array of size (timepoints,)
        '''
        if block_index + 1 > len(self.data.keys()):
            raise IndexError('block index out of range')

        block = []
        for ch in self.data.keys():
            block.append(self.data[ch][block_index])
        if len(block) == 1:
            return np.array(block[0])
        return np.stack(block)
        
    @property
    def sample_rate(self) -> np.ndarray:
        '''
        Property which returns the sample rate for all channels.
        '''
        return self.mat_dict['samplerate']