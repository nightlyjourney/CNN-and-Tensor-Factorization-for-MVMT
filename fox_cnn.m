load FOX_LabelMat
load FOX_TFIDF_Normalized
load FOX_ImageFeatureNormalizedMat

view1 = imageFeatureNormalizedMat;
view1 = view1-0.5;
% [~,Idx] = sort(sum(TFIDF_Normalized>0,2));
% wIdx = Idx(end-995:end)';
wIdx = sum(TFIDF_Normalized>0,2)>=size(TFIDF_Normalized,2)*0.01;
view2 = TFIDF_Normalized(wIdx,:);
instance_size = size(view1, 2);
fm_size = size(view1,1)*size(view2,1);
%data = sparse(instance_size,fm_size);
%for instance = 1:instance_size
%    data(instance,:) = kron(view1(:,instance)',view2(:,instance)');
%end
data = [view1; view2];
view_index = [size(view1,1),size(view1,1)+size(view2,1)];

labels = labelMat;
%labels(labels<=0) = -1;

num_inst = size(labels,1);
task_sample_size = num_inst;

sample_size = 0.7 * task_sample_size;
test_size = 0.2 * task_sample_size;
valid_size = 0.1 * task_sample_size;
v = zeros(2,4);
num_task = size(labels,2);

rndIdx = randperm(num_inst);
trainNum = sample_size;
testNum = trainNum + test_size;
validNum = testNum + valid_size;
    
trainIdx = rndIdx <= trainNum;
testIdx = rndIdx > trainNum & rndIdx <=testNum;
validIdx = rndIdx > testNum & rndIdx <= validNum;
          
train_label = labels(trainIdx,:); 
test_label = labels(testIdx,:);
valid_label = labels(validIdx,:);

train_data = data(:,trainIdx);
test_data = data(:,testIdx);
valid_data = data(:,validIdx);
v(1,:) = sum(train_label>0,1);
v(2,:) = sum(test_label>0,1);
save('fox','view_index','train_label','test_label','train_data','test_data','valid_label','valid_data');

disp(v');
