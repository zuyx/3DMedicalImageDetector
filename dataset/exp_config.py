def get_config():
    config = {}
    config['anchors'] = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    config['chanel'] = 1
    config['crop_size'] = [96, 96, 96]
    config['stride'] = 4
    config['max_stride'] = 16
    config['num_neg'] = 800
    config['th_neg'] = 0.02
    config['th_pos_train'] = 0.5
    config['th_pos_val'] = 1
    config['num_hard'] = 2
    config['bound_size'] = 12
    config['reso'] = 1
    config['sizelim'] = 2.5  # 3 #6. #mm
    config['sizelim2'] = 10  # 30
    config['sizelim3'] = 20  # 40
    config['aug_scale'] = True
    config['r_rand_crop'] = 0.3
    config['pad_value'] = 170
    config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
    config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                           'adc3bbc63d40f8761c59be10f1e504c3']
    config['fmap_size'] = [128 / 4, 128, 4, 128 / 4]
    config['top_k'] = 100
    return config

def get_deepseed_config():

    config = {}
    config['anchors'] = [5.0, 10.0, 20.]
    config['chanel'] = 1
    config['crop_size'] = [128, 128, 128]
    config['stride'] = 4
    config['max_stride'] = 16
    config['num_neg'] = 800
    config['th_neg'] = 0.02
    config['th_pos_train'] = 0.5
    config['th_pos_val'] = 1
    config['num_hard'] = 2
    config['bound_size'] = 12
    config['reso'] = 1
    config['sizelim'] = 3.  # mm, smallest nodule size
    config['sizelim2'] = 10
    config['sizelim3'] = 20
    config['aug_scale'] = True
    config['r_rand_crop'] = 0.5
    config['pad_value'] = 0
    config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
    config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                           'adc3bbc63d40f8761c59be10f1e504c3']
    config['top_k'] = 100
    return config

def get_scpm_config():
    config = {}
    config['anchors'] = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    config['chanel'] = 1
    config['crop_size'] = [96, 96, 96]
    config['stride'] = 2
    config['max_stride'] = 8
    config['num_neg'] = 800
    config['th_neg'] = 0.02
    config['th_pos_train'] = 0.5
    config['th_pos_val'] = 1
    config['num_hard'] = 2
    config['bound_size'] = 12
    config['reso'] = 1
    config['sizelim'] = 2.5  # 3 #6. #mm
    config['sizelim2'] = 10  # 30
    config['sizelim3'] = 20  # 40
    config['aug_scale'] = True
    config['r_rand_crop'] = 0.3
    config['pad_value'] = 170
    config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
    config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                           'adc3bbc63d40f8761c59be10f1e504c3']
    config['fmap_size'] = [48,48,48]
    config['top_k'] = 100
    return config