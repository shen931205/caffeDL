import os
import numpy as np
import h5py
 
def split_HDF5_val_file(h5_path):
	input_path = os.path.join(h5_path, 'val', 'val.h5')
        print input_path
	h5_file = h5py.File(input_path)
        save_path = os.path.join(h5_path, 'val')
	image_num = np.shape(h5_file['data'])[0]
	f1 = h5py.File(os.path.join(save_path, 'val1.h5'), 'w')
	f2 = h5py.File(os.path.join(save_path, 'val2.h5'), 'w')
        f3 = h5py.File(os.path.join(save_path, 'val3.h5'), 'w')
	
	f1.create_dataset('data', (12871, 3, 227, 227), dtype=np.float32)
	f1.create_dataset('label', (12871, 10), dtype=np.float32)
	f2.create_dataset('data', (12872, 3, 227, 227), dtype=np.float32)
	f2.create_dataset('label', (12872, 10), dtype=np.float32)
	f3.create_dataset('data', (12872, 3, 227, 227), dtype=np.float32)
	f3.create_dataset('label', (12872, 10), dtype=np.float32)	


	for ind in xrange(image_num):
		if ind < 12871:
                        f1['data'][ind] = h5_file['data'][ind]
                        f1['label'][ind] = h5_file['label'][ind]
		elif ind >= 12871 and ind < 25743:
                        f2['data'][ind-12871] = h5_file['data'][ind]
                        f2['label'][ind-12871] = h5_file['label'][ind]
		else:
                        f3['data'][ind-25743] = h5_file['data'][ind]
                        f3['label'][ind-25743] = h5_file['label'][ind]
		if ind % 10000 == 0 and ind != 0:
			print '{} done.'.format(ind)

	f1.close()
	f2.close()
	f3.close()
	h5_file.close()

def split_HDF5_test_file(h5_path):
        input_path = os.path.join(h5_path, 'test', 'test.h5')
        print input_path
        h5_file = h5py.File(input_path)
	save_path = os.path.join(h5_path, 'test')
	f1 = h5py.File(os.path.join(save_path, 'test1.h5'), 'w')
	f2 = h5py.File(os.path.join(save_path, 'test2.h5'), 'w')
        f3 = h5py.File(os.path.join(save_path, 'test3.h5'), 'w')
        f4 = h5py.File(os.path.join(save_path, 'test4.h5'), 'w')
	f5 = h5py.File(os.path.join(save_path, 'test5.h5'), 'w')
	f6 = h5py.File(os.path.join(save_path, 'test6.h5'), 'w')
	f7 = h5py.File(os.path.join(save_path, 'test7.h5'), 'w')
        f8 = h5py.File(os.path.join(save_path, 'test8.h5'), 'w')
        f9 = h5py.File(os.path.join(save_path, 'test9.h5'), 'w')
	f10 = h5py.File(os.path.join(save_path, 'test10.h5'), 'w')
        image_num = np.shape(h5_file['data'])[0]

        
        f1.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f1.create_dataset('label', (5792, 10), dtype=np.float32)
	f2.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f2.create_dataset('label', (5792, 10), dtype=np.float32)
	f3.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f3.create_dataset('label', (5792, 10), dtype=np.float32)
	f4.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f4.create_dataset('label', (5792, 10), dtype=np.float32)
	f5.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f5.create_dataset('label', (5792, 10), dtype=np.float32)
	f6.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f6.create_dataset('label', (5792, 10), dtype=np.float32)
	f7.create_dataset('data', (5792, 3, 227, 227), dtype=np.float32)
	f7.create_dataset('label', (5792, 10), dtype=np.float32)
	f8.create_dataset('data', (5793, 3, 227, 227), dtype=np.float32)
	f8.create_dataset('label', (5793, 10), dtype=np.float32)
	
	f9.create_dataset('data', (5793, 3, 227, 227), dtype=np.float32)
	f9.create_dataset('label', (5793, 10), dtype=np.float32)
	f10.create_dataset('data', (5793, 3, 227, 227), dtype=np.float32)
	f10.create_dataset('label', (5793, 10), dtype=np.float32)

        for ind in xrange(image_num):
		if ind < 5792:
                        f1['data'][ind] = h5_file['data'][ind]
                        f1['label'][ind] = h5_file['label'][ind]
                        
		elif ind >= 5792 and ind < 11584:
                        f2['data'][ind-5792] = h5_file['data'][ind]
                        f2['label'][ind-5792] = h5_file['label'][ind]
                        
		elif ind >= 11584 and ind < 17376:
                        f3['data'][ind-11584] = h5_file['data'][ind]
                        f3['label'][ind-11584] = h5_file['label'][ind]
                        
                elif ind >= 17376 and ind < 23168:
                        f4['data'][ind-17376] = h5_file['data'][ind]
                        f4['label'][ind-17376] = h5_file['label'][ind]
                        
                elif ind >= 23168 and ind < 28960:
                        f5['data'][ind-23168] = h5_file['data'][ind]
                        f5['label'][ind-23168] = h5_file['label'][ind]
                        
                elif ind >= 28960 and ind < 34752:
                        f6['data'][ind-28960] = h5_file['data'][ind]
                        f6['label'][ind-28960] = h5_file['label'][ind]
                        
                elif ind >= 34752 and ind < 40544:
                        f7['data'][ind-34752] = h5_file['data'][ind]
                        f7['label'][ind-34752] = h5_file['label'][ind]
                        
                elif ind >= 40544 and ind < 46337:
                        f8['data'][ind-40544] = h5_file['data'][ind]
                        f8['label'][ind-40544] = h5_file['label'][ind]
                        
                elif ind >= 46337 and ind < 52130:
                        f9['data'][ind-46337] = h5_file['data'][ind]
                        f9['label'][ind-46337] = h5_file['label'][ind]
                        print ind
                else:
                        print '-----------------f10'
                        f10['data'][ind-52130] = h5_file['data'][ind]
                        f10['label'][ind-52130] = h5_file['label'][ind]
                        
                        
		if ind % 10000 == 0 and ind != 0:
			print '{} done.'.format(ind)

	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()
	f6.close()
	f7.close()
	f8.close()
	f9.close()
	f10.close()
	h5_file.close()

def split_HDF5_train_file(h5_path):
        input_path = os.path.join(h5_path, 'train', 'train.h5')
        print input_path
	h5_file = h5py.File(input_path)
	save_path = os.path.join(h5_path, 'train')
        f1 = h5py.File(os.path.join(save_path, 'train1.h5'), 'w')
	f2 = h5py.File(os.path.join(save_path, 'train2.h5'), 'w')
        f3 = h5py.File(os.path.join(save_path, 'train3.h5'), 'w')
        f4 = h5py.File(os.path.join(save_path, 'train4.h5'), 'w')
	f5 = h5py.File(os.path.join(save_path, 'train5.h5'), 'w')
	f6 = h5py.File(os.path.join(save_path, 'train6.h5'), 'w')
	f7 = h5py.File(os.path.join(save_path, 'train7.h5'), 'w')
        f8 = h5py.File(os.path.join(save_path, 'train8.h5'), 'w')
        f9 = h5py.File(os.path.join(save_path, 'train9.h5'), 'w')
	f10 = h5py.File(os.path.join(save_path, 'train10.h5'), 'w')
        image_num = np.shape(h5_file['data'])[0]

        
        f1.create_dataset('data', (9653, 3, 227, 227), dtype=np.float32)
	f1.create_dataset('label', (9653, 10), dtype=np.float32)
	f2.create_dataset('data', (9653, 3, 227, 227), dtype=np.float32)
	f2.create_dataset('label', (9653, 10), dtype=np.float32)
	f3.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f3.create_dataset('label', (9654, 10), dtype=np.float32)
	f4.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f4.create_dataset('label', (9654, 10), dtype=np.float32)
	f5.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f5.create_dataset('label', (9654, 10), dtype=np.float32)
	f6.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f6.create_dataset('label', (9654, 10), dtype=np.float32)
	f7.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f7.create_dataset('label', (9654, 10), dtype=np.float32)
	f8.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f8.create_dataset('label', (9654, 10), dtype=np.float32)
	
	f9.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f9.create_dataset('label', (9654, 10), dtype=np.float32)
	f10.create_dataset('data', (9654, 3, 227, 227), dtype=np.float32)
	f10.create_dataset('label', (9654, 10), dtype=np.float32)

        for ind in xrange(image_num):
		if ind < 9653:
                        f1['data'][ind] = h5_file['data'][ind]
                        f1['label'][ind] = h5_file['label'][ind]
                        
		elif ind >= 9653 and ind < 19306:
                        f2['data'][ind-9653] = h5_file['data'][ind]
                        f2['label'][ind-9653] = h5_file['label'][ind]
                        
		elif ind >= 19306 and ind < 28960:
                        f3['data'][ind-19306] = h5_file['data'][ind]
                        f3['label'][ind-19306] = h5_file['label'][ind]
                        
                elif ind >= 28960 and ind < 38614:
                        f4['data'][ind-28960] = h5_file['data'][ind]
                        f4['label'][ind-28960] = h5_file['label'][ind]
                        
                elif ind >= 38614 and ind < 48268:
                        f5['data'][ind-38614] = h5_file['data'][ind]
                        f5['label'][ind-38614] = h5_file['label'][ind]
                        
                elif ind >= 48268 and ind < 57922:
                        f6['data'][ind-48268] = h5_file['data'][ind]
                        f6['label'][ind-48268] = h5_file['label'][ind]
                        
                elif ind >= 57922 and ind < 67576:
                        f7['data'][ind-57922] = h5_file['data'][ind]
                        f7['label'][ind-57922] = h5_file['label'][ind]
                        
                elif ind >= 67576 and ind < 77230:
                        f8['data'][ind-67576] = h5_file['data'][ind]
                        f8['label'][ind-67576] = h5_file['label'][ind]
                        
                elif ind >= 77230 and ind < 86884:
                        f9['data'][ind-77230] = h5_file['data'][ind]
                        f9['label'][ind-77230] = h5_file['label'][ind]
                        print ind
                else:
                        print '-----------------f10'
                        f10['data'][ind-87884] = h5_file['data'][ind]
                        f10['label'][ind-87884] = h5_file['label'][ind]
                        
                        
		if ind % 10000 == 0 and ind != 0:
			print '{} done.'.format(ind)

	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()
	f6.close()
	f7.close()
	f8.close()
	f9.close()
	f10.close()
	h5_file.close()
        

if __name__ == '__main__':    
    path = '/home/u514/DTask/data/AVA'
    split_HDF5_val_file(path)
    split_HDF5_test_file(path)
    split_HDF5_train_file(path)
