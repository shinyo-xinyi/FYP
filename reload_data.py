import pickle
from librosa.util import find_files
import scipy.io as sio
import glob, os

access_type = "LA"
# # on air station gpu
path_to_mat = 'features/ASVspoof2019/'+access_type+'/Features_MATLAB/'
path_to_audio = 'F:/FYP/dataset/'+access_type+'/ASVspoof2019_'+access_type+'_'
path_to_features = 'features/ASVspoof2019/'+access_type+'/Features_Python/'

def reload_data(path_to_features, part):
    # matfiles = find_files(path_to_mat + part + '/', ext='mat') # origin
    matfiles = glob.glob(os.path.join(path_to_mat + part, '*_aug.mat')) # augmented
    for i in range(len(matfiles)):
        # key = matfiles[i][len(path_to_mat) + len(part) + 21:-4] # origin
        key = matfiles[i][len(path_to_mat) + len(part) + 1 :-4] # augmented
        # print(key)
        lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
        with open(path_to_features + part +'/'+ key + '.pkl', 'wb') as handle2:
            pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # reload_data(path_to_features, 'train')
    # reload_data(path_to_features, 'dev')
    reload_data(path_to_features, 'eval')
