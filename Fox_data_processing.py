import scipy.io as scio

#for i in range(4)
dataFile = '/home/dinghao/CNN MTMV/fox.mat'
data = scio.loadmat(dataFile)
view_index = data['view_index']
train_label = (data['train_label']).toarray()
test_label = (data['test_label']).toarray()
train_data = data['train_data']
test_data = data['test_data']
valid_label = (data['valid_label']).toarray()
valid_data = data['valid_data']

train_instance_num = train_data.shape[1]
valid_instance_num = valid_data.shape[1]
test_instance_num = test_data.shape[1]

#pre-process data
def data_preprocess(instance_num, data, view_index):
   view1_index = int(view_index[0,0])
   view2_index = int(view_index[0,1])
   input_data = lil_matrix((instance_num, view1_index*(view2_index-view1_index)),dtype=np.float32)
   for i in xrange(instance_num):
      instance = data[:,i]
      view1 = instance.tocsr()[0:view1_index,:]
      view2 = instance.tocsr()[view1_index:view2_index,:]
      matrix = view1*view2.transpose()
      input_data[i,:] = (matrix.tolil()).reshape((1,view1_index*(view2_index-view1_index)))
   return input_data.toarray()
 
print 'Data-Preprocessing...'
train_data = data_preprocess(train_instance_num, train_data, view_index)
print 'Train_data Preprocess Done'
valid_data = data_preprocess(valid_instance_num, valid_data, view_index)
print 'Valid_data Preprocess Done'
test_data = data_preprocess(test_instance_num, test_data, view_index)
print 'Test_data Preprocess Done'

scio.savemat('fox_data.mat', {'train_data': train_data,'train_label': train_label,'valid_data': valid_data,'valid_label': valid_label,'test_data':test_data,'test_label':test_label,'view_index':view_index})
