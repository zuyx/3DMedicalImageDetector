
data_root_dir = '/home/ren/zyx/datasets/'
project_root_dir = '/home/ren/zyx/Lesion/NoduleDetector/detectorv2/'

config = {'luna_raw': data_root_dir + 'luna16/luna16',
          'luna_segment': data_root_dir + 'luna16/seg-lungs-LUNA16/',

          'luna_data': data_root_dir + 'luna16/ReName_DATA',
          'preprocess_result_path': data_root_dir + 'luna16/procdata/',
          'debug_patient_ids': 'dataset/data/luna_debug.npy',
          'train_patient_ids':  'dataset/data/luna_train.npy',
          'test_patient_ids':  'dataset/data/luna_test.npy',
          'train_debug_patients_ids':'dataset/data/luna_train_debug.npy',
          'luna_abbr': data_root_dir + 'luna16/labels/shorter.csv',  # './detector/labels/shorter.csv',
          'luna_label': data_root_dir + 'luna16/labels/annos.csv',
          'preprocessing_backend': 'python'
          }

if __name__ == '__main__':
    import numpy as np
    import os
    print(os.getcwd())
    pids = np.load('luna_train.npy')
    pids = pids[:100]
    file = "luna_train_debug.npy"
    if not os.path.exists(file):
        np.save(file,pids)
    debug_ids = np.load(file)
    print(debug_ids)

