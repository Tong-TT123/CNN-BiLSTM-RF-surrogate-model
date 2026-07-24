function [cnn_feature_model,layer_get,train_x_feature_label_norm1,vaild_x_feature_label_norm1,test_x_feature_label_norm1]=CNN_feature_extract(num_feature,train_x_feature_label_norm,train_y_feature_label_norm,vaild_x_feature_label_norm,test_x_feature_label_norm,augment_feature_method)
% CNN 回归特征提取
   epoch_set=120;    %神经网络隐藏层
   train_x_feature_label=train_x_feature_label_norm;
   vaild_x_feature_label=vaild_x_feature_label_norm;
   test_x_feature_label=test_x_feature_label_norm;

if strcmp(augment_feature_method,'CNN')==1
    p_train1=reshape(train_x_feature_label_norm',size(train_x_feature_label,2),1,1,size(train_x_feature_label,1));

    p_vaild1=reshape(vaild_x_feature_label_norm',size(vaild_x_feature_label,2),1,1,size(vaild_x_feature_label,1));

    p_test1=reshape(test_x_feature_label_norm',size(test_x_feature_label,2),1,1,size(test_x_feature_label,1));

    layers = [             % 建立输入层
        imageInputLayer([ size(train_x_feature_label,2) 1 1])%%2D-CNN
        convolution2dLayer([2,1],8, "Name", "conv1") %卷积 [2,1] filterSize 卷积核大小, 8 numFilters  卷积核数量 默认 stride [1  1] and padding [0  0  0  0]
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([2 1],'Stride',1)   %池化 poolsize [2 1]
        convolution2dLayer([2,1],8, "Name", "conv2")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([2 1],'Stride',1, "Name", "pool2")
        fullyConnectedLayer(num_feature, "Name", "out1")    %num_feature  特征提取维度
        fullyConnectedLayer(1)
        regressionLayer];

    options = trainingOptions('adam', ...
        'Shuffle','every-epoch',...
        'MaxEpochs',epoch_set, ...,
        'InitialLearnRate',0.001,...
        'Plots','training-progress');

    Mdl= trainNetwork(p_train1, train_y_feature_label_norm, layers, options);

    cnn_feature_model=Mdl;

    layer = 'out1';  %特征提取层
    layer_get=layer;
    train_x_feature_label_norm1 = double(activations(Mdl,p_train1,layer,'OutputAs','rows')); %特征进行拓展行输出
    vaild_x_feature_label_norm1  = double(activations(Mdl,p_vaild1, layer,'OutputAs','rows'));
    test_x_feature_label_norm1  = double(activations(Mdl,p_test1, layer,'OutputAs','rows'));

elseif strcmp(augment_feature_method,'LSTM')==1||strcmp(augment_feature_method,'BiLSTM')==1||strcmp(augment_feature_method,'LSTM-attention')==1

    for i = 1: size(train_x_feature_label,1)      %修改输入变成元胞形式
        p_train1{i, 1} = (train_x_feature_label_norm(i,:))';
    end
    for i = 1 : size(test_x_feature_label,1)
        p_test1{i, 1}  = (test_x_feature_label_norm(i,:))';
    end
    for i = 1 : size(vaild_x_feature_label,1)
        p_vaild1{i, 1}  = (vaild_x_feature_label_norm(i,:))';
    end

if strcmp(augment_feature_method,'LSTM')==1
    layers = [sequenceInputLayer(size(train_x_feature_label_norm,2))               % 建立输入层
    lstmLayer(32, 'OutputMode', 'last')      % LSTM层
    reluLayer                                               % Relu激活层
    fullyConnectedLayer(num_feature, "Name", "out1")          % 全连接层
    dropoutLayer(0.2)                                 % 防止过拟合
    fullyConnectedLayer(1)          % 全连接层
    regressionLayer];    

elseif strcmp(augment_feature_method,'BiLSTM')==1
    layers = [sequenceInputLayer(size(train_x_feature_label_norm,2))               % 建立输入层
    bilstmLayer(32, 'OutputMode', 'last')      % LSTM层
    reluLayer                                               % Relu激活层
    fullyConnectedLayer(num_feature, "Name", "out1")          % 全连接层
    dropoutLayer(0.2)                                 % 防止过拟合
    fullyConnectedLayer(1)          % 全连接层
    regressionLayer];    

elseif strcmp(augment_feature_method,'LSTM-attention')==1
    layers = [sequenceInputLayer(size(train_x_feature_label_norm,2))               % 建立输入层
    lstmLayer(32, 'OutputMode', 'last')      % LSTM层
    selfAttentionLayer(2,4)
    reluLayer                                               % Relu激活层
    fullyConnectedLayer(num_feature, "Name", "out1")          % 全连接层
    dropoutLayer(0.2)                                 % 防止过拟合
    fullyConnectedLayer(1)          % 全连接层
    regressionLayer];    
    
end
options = trainingOptions('adam', ...
    'Shuffle','every-epoch',...
    'MaxEpochs',epoch_set, ...,
    'InitialLearnRate',0.001,...
    'ValidationFrequency',20, ...
    'Plots','training-progress');

[Mdl,~] = trainNetwork(p_train1, train_y_feature_label_norm, layers, options);
cnn_feature_model=Mdl;

layer = 'out1';  %特征提取层
layer_get=layer;
train_x_feature_label_norm1 = double(activations(Mdl,p_train1,layer,'OutputAs','rows')); %特征进行拓展行输出
vaild_x_feature_label_norm1  = double(activations(Mdl,p_vaild1, layer,'OutputAs','rows'));
test_x_feature_label_norm1  = double(activations(Mdl,p_test1, layer,'OutputAs','rows'));

else

    for i = 1: size(train_x_feature_label,1)      %修改输入变成元胞形式
        p_train1{i, 1} = (train_x_feature_label_norm(i,:))';
    end
    for i = 1 : size(test_x_feature_label,1)
        p_test1{i, 1}  = (test_x_feature_label_norm(i,:))';
    end
    for i = 1 : size(vaild_x_feature_label,1)
        p_vaild1{i, 1}  = (vaild_x_feature_label_norm(i,:))';
    end


    attention_head=[2,4]; %注意力机制的头head 和键 keys
    maxPosition=32;
    layers = [
        sequenceInputLayer(length(p_train1{1,1}),Name="input")
        positionEmbeddingLayer(length(p_train1{1,1}),maxPosition,Name="pos-emb")
        additionLayer(2,Name="add")
        selfAttentionLayer(attention_head(1),attention_head(2))
        dropoutLayer(0.2)                                 % 防止过拟合
        selfAttentionLayer(attention_head(1),attention_head(2))
        dropoutLayer(0.2);
        lstmLayer(16,'OutputMode','last')% 回归层
        fullyConnectedLayer(num_feature, "Name", "out1") 
        dropoutLayer(0.2);
        fullyConnectedLayer(1)
        regressionLayer];

    lgraph = layerGraph(layers);
    layers = connectLayers(lgraph,"input","add/in2");

    options = trainingOptions('adam', ...
        'Shuffle','every-epoch',...
        'MaxEpochs',epoch_set, ...,
        'InitialLearnRate',0.001,...
        'ValidationFrequency',20, ...
        'Plots','training-progress');

    [Mdl,~] = trainNetwork(p_train1, train_y_feature_label_norm, layers, options);
    cnn_feature_model=Mdl;

    layer = 'out1';  %特征提取层
    layer_get=layer;
    train_x_feature_label_norm1 = double(activations(Mdl,p_train1,layer,'OutputAs','rows')); %特征进行拓展行输出
    vaild_x_feature_label_norm1  = double(activations(Mdl,p_vaild1, layer,'OutputAs','rows'));
    test_x_feature_label_norm1  = double(activations(Mdl,p_test1, layer,'OutputAs','rows'));
end
