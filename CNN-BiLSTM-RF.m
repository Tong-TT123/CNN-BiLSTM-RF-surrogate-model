clc;clear;close all;	
load('R_16_Jul_2026_10_46_51.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
     end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
 %% 数值类型数据处理	
if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
    num_NAN__COL=sum(data_NAN1');	
    index_NAN1=num_NAN__COL>0;	
    index_double2_index=1:size(data_numshuju,1);	
    index_double2_index(index_NAN1==1)=[];	
    data_numshuju2=data_numshuju1(index_double2_index,:);	
    index_need_last=index_double1;	
 else	
    index_need_last=[];	
    data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
    if j==length(index_char)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
 data=data_all_last;	
 data_biao_all=data1.Properties.VariableNames;	
 for j=1:length(label_all_last)	
    data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
 end	
	
% 异常值检测	
	
 unique_index_ab=G_out_data.unique_index_ab; 	
 data(:,unique_index_ab)=[];	
 label_all_last(unique_index_ab)=[];	
 data_biao1=data_biao; data_biao1(unique_index_ab)=[]; 	
	
detect_label=5;   [data,index_ab_unique]=abnorm_detect(data,data_biao1,detect_label);	
	
%%  特征处理 特征选择或者降维	
	
 A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=1:(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
	
	
	
num_feature=G_out_data. num_feature; 	
augment_feature_method=G_out_data. augment_feature_method; 	
	
[cnn_feature_model,layer_get,train_x_feature_label_norm1,vaild_x_feature_label_norm1,test_x_feature_label_norm1]=CNN_feature_extract(num_feature,train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,test_x_feature_label_norm,augment_feature_method);  	
  	
train_x_feature_label_norm=[train_x_feature_label_norm,train_x_feature_label_norm1];  	
vaild_x_feature_label_norm=[vaild_x_feature_label_norm,vaild_x_feature_label_norm1];  	
test_x_feature_label_norm=[test_x_feature_label_norm,test_x_feature_label_norm1];  	
x_sig(size(x_feature_label,2)+1:size(x_feature_label,2)+num_feature)=1; 	
x_mu(size(x_feature_label,2)+1:size(x_feature_label,2)+num_feature)=0; 	
	
%% 算法处理块	
	
	
	
t1=clock;	
disp('优化CBiLSTM-RF回归')	
  	
p_train1=reshape(train_x_feature_label_norm',size(train_x_feature_label_norm,2),1,1,size(train_x_feature_label,1));	
 	
 	
p_vaild1=reshape(vaild_x_feature_label_norm',size(vaild_x_feature_label_norm,2),1,1,size(vaild_x_feature_label,1)); 	
 	
  	
p_test1=reshape(test_x_feature_label_norm',size(test_x_feature_label_norm,2),1,1,size(test_x_feature_label,1)); 	
  	
	
	
 [Model_CBiLSTM,~,fitness,Loss,pop]=optimize_fitrCNN_BILSTM_att(p_train1,train_y_feature_label_norm,p_vaild1,vaild_y_feature_label_norm,num_pop,num_iter,method_mti,max_epoch,min_batchsize,attention_label,attention_head);	
 [Model_RF] = optimize_fitrtreebag1(train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,vaild_y_feature_label_norm,num_pop,num_iter,method_mti);   	
	
y_train_predict_norm_RF= predict(Model_RF,train_x_feature_label_norm); 	
y_train_predict_norm_CBiLSTM= predict(Model_CBiLSTM,p_train1,'MiniBatchSize',min_batchsize);	
y_vaild_predict_norm_RF= predict(Model_RF,vaild_x_feature_label_norm);	
y_vaild_predict_norm_CBiLSTM= predict(Model_CBiLSTM,p_vaild1,'MiniBatchSize',min_batchsize); 	
y_test_predict_norm_RF= predict(Model_RF,test_x_feature_label_norm); 	
y_test_predict_norm_CBiLSTM= predict(Model_CBiLSTM,p_test1,'MiniBatchSize',min_batchsize); 	
 	
errors_CBiLSTM=sum(abs(y_vaild_predict_norm_CBiLSTM*y_sig+y_mu-vaild_y_feature_label))/length(vaild_y_feature_label) ;  	
errors_RF=sum(abs(y_vaild_predict_norm_RF*y_sig+y_mu-vaild_y_feature_label))/length(vaild_y_feature_label) ;  	
  	
 if abs((errors_RF-errors_CBiLSTM)/max(errors_RF,errors_CBiLSTM))<0.1 	
      quan1=errors_CBiLSTM/(errors_RF+errors_CBiLSTM);quan2=errors_RF/(errors_RF+errors_CBiLSTM); 	
else 	
     if(errors_RF>errors_CBiLSTM)	
       quan1=0;quan2=1;     	
     else  	
       quan1=1;quan2=0; 	
     end 	
 end	
  	
Mdl{1,1}=Model_RF; Mdl{1,2}=Model_CBiLSTM; Mdl{1,3}=[quan1,quan2]; 	
y_train_predict_norm=quan1*y_train_predict_norm_RF+quan2*y_train_predict_norm_CBiLSTM; 	
y_vaild_predict_norm=quan1*y_vaild_predict_norm_RF+quan2*y_vaild_predict_norm_CBiLSTM; 	
y_test_predict_norm=quan1*y_test_predict_norm_RF+quan2*y_test_predict_norm_CBiLSTM; 	
y_test_predict_CBiLSTM_RF=y_test_predict_norm*y_sig+y_mu; 	
y_test_predict_RF=y_test_predict_norm_RF*y_sig+y_mu; 	
y_test_predict_CBiLSTM=y_test_predict_norm_CBiLSTM*y_sig+y_mu;  	
disp(['RF验证集平均绝对误差MAE：',num2str(errors_RF)])      	
disp(['CBiLSTM验证集平均绝对误差MAE：',num2str(errors_CBiLSTM)])  	
disp(['CBiLSTM-RF结合权重：',num2str([quan1,quan2])])  	
CBiLSTM_RF_MAE=sum(abs(y_test_predict_CBiLSTM_RF-test_y_feature_label))/length(test_y_feature_label) ;  	
disp(['CBiLSTM-RF测试集平均绝对误差MAE：',num2str(CBiLSTM_RF_MAE)])  ;	
RF_MAE=sum(abs(y_test_predict_RF-test_y_feature_label))/length(test_y_feature_label) ;  	
disp(['RF测试集平均绝对误差MAE：',num2str(RF_MAE)]) 	
CBiLSTM_MAE=sum(abs(y_test_predict_CBiLSTM-test_y_feature_label))/length(test_y_feature_label) ;	
disp(['CBiLSTM测试集平均绝对误差MAE：',num2str(CBiLSTM_MAE)])   	
 	
t2=clock; 	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));  	
	
 analyzeNetwork(Model_CBiLSTM)  	
   	
subplot(2, 1, 1)	
plot(1 : length(Loss.TrainingRMSE), Loss.TrainingRMSE, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('均方根误差');legend('训练集均方根误差');title ('训练集均方根误差曲线');grid;set(gcf,'color','w')	
	
subplot(2, 1, 2)	
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('损失函数');legend('训练集损失值');title ('训练集损失函数曲线');grid;set(gcf,'color','w')	
	
	
	
 y_train_predict=y_train_predict_norm*y_sig+y_mu;  %反标准化操作 	
 y_vaild_predict=y_vaild_predict_norm*y_sig+y_mu; 	
 y_test_predict=y_test_predict_norm*y_sig+y_mu; 	
 train_y=train_y_feature_label; disp('***************************************************************************************************************')   	
 train_MAE=sum(abs(y_train_predict-train_y))/length(train_y) ; disp(['训练集平均绝对误差MAE：',num2str(train_MAE)])	
 train_MAPE=sum(abs((y_train_predict-train_y)./train_y))/length(train_y); disp(['训练集平均相对误差MAPE：',num2str(train_MAPE)])	
 train_MSE=(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方误差MSE：',num2str(train_MSE)]) 	
 train_RMSE=sqrt(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
 train_R2= 1 - (norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp(['训练集R方系数R2：',num2str(train_R2)]) 	
 vaild_y=vaild_y_feature_label;disp('***************************************************************************************************************')	
 vaild_MAE=sum(abs(y_vaild_predict-vaild_y))/length(vaild_y) ; disp(['验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
 vaild_MAPE=sum(abs((y_vaild_predict-vaild_y)./vaild_y))/length(vaild_y); disp(['验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
 vaild_MSE=(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方误差MSE：',num2str(vaild_MSE)])     	
 vaild_RMSE=sqrt(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
 vaild_R2= 1 - (norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);    disp(['验证集R方系数R2:  ',num2str(vaild_R2)])			
 test_y=test_y_feature_label;disp('***************************************************************************************************************');   	
 test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ; disp(['测试集平均绝对误差MAE：',num2str(test_MAE)])        	
 test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y); disp(['测试集平均相对误差MAPE：',num2str(test_MAPE)])	
 test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方误差MSE：',num2str(test_MSE)]) 	
 test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
 test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp(['测试集R方系数R2：',num2str(test_R2)]) 	
	
	
 test_y1=[vaild_y_feature_label;test_y_feature_label];  y_test_predict1=[y_vaild_predict;y_test_predict];;disp('验证集+测试集***************************************************************************************************************');   	
 test1_MAE=sum(abs(y_test_predict1-test_y1))/length(test_y1) ; disp(['验证集+测试集平均绝对误差MAE：',num2str(test1_MAE)])        	
 test1_MAPE=sum(abs((y_test_predict1-test_y1)./test_y1))/length(test_y1); disp(['验证集+测试集平均相对误差MAPE：',num2str(test1_MAPE)])	
 test1_MSE=(sum(((y_test_predict1-test_y1)).^2)/length(test_y1)); disp(['验证集+测试集均方误差MSE：',num2str(test1_MSE)]) 	
 test1_RMSE=sqrt(sum(((y_test_predict1-test_y1)).^2)/length(test_y1)); disp(['验证集+测试集均方根误差RMSE：',num2str(test1_RMSE)]) 	
 test1_R2= 1 -sum((test_y1 - y_test_predict1).^2) /sum((test_y1 - mean(test_y1)).^2);   disp(['验证集+测试集R方系数R2：',num2str(test1_R2)])	
 disp(['算法运行时间Time: ',num2str(Time)])	
