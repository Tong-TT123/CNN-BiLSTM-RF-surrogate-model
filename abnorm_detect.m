function [y,index_ab_unique]=abnorm_detect(x1,label_index,detect_label)
% detect_label=1 3sigma  detect_label=2 四分位   detect_label=3 zscore
% detect_label=4  孤立森林
rng(1)
x=zscore(x1);   %都先进行标准化处理
if detect_label==1
figure
index_ab_all=[];
for i=1:size(x,2)
    x_mean=mean(x(:,i));
    x_std=std(x(:,i));
    index_ab=find(abs(x(:,i)-x_mean)>3*x_std);  %异常值
    if size(index_ab,2)>1
        index_ab=index_ab';
    end
    index_ab_all=[index_ab_all;index_ab];
    index_nor=1:size(x,1);
    index_nor(index_ab)=[];
    scatter(i*ones(1,length(index_nor)),x(index_nor,i),'b')
    hold on
    scatter(i*ones(1,length(index_ab)),x(index_ab,i),'r')
    hold on
end
index_ab_unique=unique(index_ab_all);
disp("存在异常值的行为：")
disp(index_ab_unique')

xticks(1:size(x,2))
xticklabels(label_index);
AA=x1;
AA(index_ab_unique,:)=[];
y=AA;
elseif detect_label==2
figure;
boxplot(x)
xticks(1:size(x,2))
xticklabels(label_index);
index_ab_all=[];
for i=1:size(x,2)
    Q=quantile(x(:,i),[ 0.25  0.75 ]);
    q1=Q(1)-1.5*(Q(2)-Q(1));
    q2=Q(2)+1.5*(Q(2)-Q(1));
    index_ab=find(x(:,i)<q1|x(:,i)>q2);
   index_ab_all=[index_ab_all;index_ab];
end
index_ab_unique=unique(index_ab_all);
disp("存在异常值的行为：")
disp(index_ab_unique')

AA=x1;
AA(index_ab_unique,:)=[];
y=AA;

elseif detect_label==3
figure
index_ab_all=[];
for i=1:size(x,2)
%     x_mean=mean(x(:,i));
%     x_std=std(x(:,i));
    index_ab=find((abs(x(:,i))-3)>0);  %异常值
    if size(index_ab,2)>1
        index_ab=index_ab';
    end
    index_ab_all=[index_ab_all;index_ab];
    index_nor=1:size(x,1);
    index_nor(index_ab)=[];
    scatter(i*ones(1,length(index_nor)),x(index_nor,i),'b')
    hold on
    scatter(i*ones(1,length(index_ab)),x(index_ab,i),'r')
    hold on
end
index_ab_unique=unique(index_ab_all);
disp("存在异常值的行为：")
disp(index_ab_unique')

xticks(1:size(x,2))
xticklabels(label_index);
AA=x1;
AA(index_ab_unique,:)=[];
y=AA;

elseif detect_label==4
    [Mdl,tf,scores] = iforest(x,ContaminationFraction=0.05);
    histogram(scores)
    xline(Mdl.ScoreThreshold,"r-",["Threshold" Mdl.ScoreThreshold])
    index_ab_unique=find(tf==1);
    disp("存在异常值的行为：")
    disp(index_ab_unique')
    AA=x1;
    AA(index_ab_unique,:)=[];
    y=AA;
    T = tsne(x, Standardize = true);

    %  绘制可视化结果
    figure
    gscatter(T(:, 1), T(:, 2), tf, "kr", [], 8, "off")
    legend("正常值", "离群值")
    title("isolation forest")
    set(gcf,'color','w')
    grid on
    xlabel('dimension1')
    ylabel('dimension2')

 elseif detect_label==5

% 异常值\缺失值处理
%主要对缺失值进行插值处理
data1=fillmissing(x1,'linear'); %线性插值
% data1=fillmissing(data,'spline'); %三次样条插值
% data1=fillmissing(data,'movmedian',10); %移动平均插值
% 离群值处理
%识别离群点并通过插值更正，离群值定义为偏离均值超过三倍标准差的值
[B_data,TF,L,U,C] = filloutliers(data1(:,:),"clip","movmedian",15);%滑动窗线性插值
data_new = B_data;

figure
index_set=1:ceil(size(data1,1)/10);
plot(data1(index_set,end))
hold on
plot(B_data(index_set,end),"o-")
hold on
legend("Original Data","Filled Data")
% 找出最相关的数据列/进行数据判别
data_new=zscore(data_new); 

cor_R2 = corrcoef(data_new);
cor_R2_end=cor_R2(end,:);
cor_R2_end(end)=[];
[~,index_max]=max(cor_R2_end);

% 对每一段进行划分

index_abnorm1=find(data_new(:,index_max)>(mean(data_new(:,index_max))+3*std(data_new(:,index_max))));
index_abnorm2=find(data_new(:,index_max)<(mean(data_new(:,index_max))-3*std(data_new(:,index_max))));
index_abnorm_all1=[index_abnorm1;index_abnorm2];
data_new(index_abnorm_all1,:)=[];
fitPoints = [data_new(:,index_max) data_new(:,end)];  %对最相关的因素和输出y进行数据拟合

nbin=10; %将X划分为10段

xpoint_bin=linspace(min(data_new(:,index_max)),max(data_new(:,index_max)),10);

fitFcn = polyfit(fitPoints(:,1),fitPoints(:,2),2); % 进行二次项拟合，也可以通过实际情况进行

ypoint_bin = polyval(fitFcn,xpoint_bin);

figure
scatter(fitPoints(:,1),fitPoints(:,2))
hold on
plot(xpoint_bin,ypoint_bin,'--')
xlabel(label_index{1,index_max})
ylabel(label_index{1,length(label_index)})
% 对多个区间段分别进行孤立森林异常识别，以异常点进行边界绘制
ContaminationFraction_rate=0.05;
if size(data_new,1)>50000
    ContaminationFraction_rate=0.02;
elseif size(data_new,1)>80000
    ContaminationFraction_rate=0.01;
end

norm_index_all_upper=[];
norm_index_all_lower=[];
norm_index_all=[];
abnorm_index_all=[];
for i =1:nbin-1
% for i =1
    index=find((fitPoints(:,1)>xpoint_bin(i))&(fitPoints(:,1)<xpoint_bin(i+1))); %取出该段数据
    if (length(index)>size(x,1)/nbin)
        ContaminationFraction_rate1=ContaminationFraction_rate/((length(index)/size(x,1))/(1/nbin));
    else
        ContaminationFraction_rate1=ContaminationFraction_rate;
    end

    y_data=fitPoints(index,end);
    mean_line_data=(ypoint_bin(i)+ypoint_bin(i+1))/2;
    [forest, tf_forest, s_forest] = iforest(y_data, ContaminationFraction = ContaminationFraction_rate1); %设置0.02的数据划分比例
    index1=index;
    index_abnorm=index1((tf_forest==1));

    data_index_abnorm=fitPoints(index_abnorm,end);

    index_abnorm_upper=index_abnorm(data_index_abnorm>mean_line_data);
    index_abnorm_lower=index_abnorm(data_index_abnorm<mean_line_data);

    data_index_abnorm_up=fitPoints(index_abnorm_upper,end);

    index_abnorm_upper1=index_abnorm_upper(data_index_abnorm_up>prctile(data_index_abnorm_up, 50));

    index_norm_upper1=index_abnorm_upper(data_index_abnorm_up<prctile(data_index_abnorm_up, 50));

    index_abnorm=[index_abnorm_lower;index_abnorm_upper1];

    index1((tf_forest==1))=[];

    abnorm_index_all=[abnorm_index_all;index_abnorm];

    index_norm=[index1;index_norm_upper1];

    norm_index_all=[norm_index_all;index_norm];

    index_norm_upper=index1(fitPoints(index1,end)>prctile(fitPoints(index1,end), 99));
    index_norm_lower=index1(fitPoints(index1,end)<prctile(fitPoints(index1,end), 2));

    norm_index_all_upper=[norm_index_all_upper;index_norm_upper];
    norm_index_all_lower=[norm_index_all_lower;index_norm_lower];
    % abnorm_cell{1,i}=find();
end
% 找出缺失数据和正常数据



norm_index_up_data=[fitPoints(norm_index_all_upper,1) fitPoints(norm_index_all_upper,end)];

norm_index_lower_data=[fitPoints(norm_index_all_lower,1) fitPoints(norm_index_all_lower,end)];

fitFcn_low = polyfit(norm_index_lower_data(:,1),norm_index_lower_data(:,2),2); % 进行二次项拟合，也可以通过实际情况进行

ypoint_bin_low = polyval(fitFcn_low,xpoint_bin);

fitFcn_up = polyfit(norm_index_up_data(:,1),norm_index_up_data(:,2),2); % 进行二次项拟合，也可以通过实际情况进行

ypoint_bin_up = polyval(fitFcn_up,xpoint_bin);


%
input_norm_data=B_data;
input_norm_data(index_abnorm_all1,:)=[];
% input_norm_data(abnorm_index_all,:)=[];
index_set=1:size(fitPoints,1);
% index_set(abnorm_index_all1)=[];

input_norm_data1=input_norm_data;

x_label_get=fitPoints(:,1);
y_label_get=fitPoints(:,2);

y_label_get_lower=polyval(fitFcn_low,x_label_get);
y_label_get_upper=polyval(fitFcn_up,x_label_get);

index_up_ubnorm=find(y_label_get>y_label_get_upper+0.3);
index_low_ubnorm=find(y_label_get<y_label_get_lower-0.2);
data_ubnorm2=[index_up_ubnorm;index_low_ubnorm];
abnorm_index=index_set(data_ubnorm2);

abnorm_index_all1=unique([abnorm_index']);
abnorm_index_data=[fitPoints(abnorm_index_all1,1) fitPoints(abnorm_index_all1,end)];

input_norm_data1(abnorm_index_all1,:)=[];
y=input_norm_data1;

figure
scatter(fitPoints(:,1),fitPoints(:,2),'Color',[0.2353    0.5176    0.7725])
hold on
plot(xpoint_bin,ypoint_bin,'--','LineWidth',1.5)
hold on
scatter(abnorm_index_data(:,1),abnorm_index_data(:,2),'Color',[0.9725    0.6039    0.1922])
hold on
% scatter(input_norm_data(data_ubnorm2,index_max),input_norm_data(data_ubnorm2,end),'Color',[0.9725    0.6039    0.1922])
% hold on
plot(xpoint_bin,ypoint_bin_low,'-','LineWidth',1.5,'Color',[0.9725    0.6039    0.1922])
hold on
plot(xpoint_bin,ypoint_bin_up,'-','LineWidth',1.5,'Color',[0.9725    0.6039    0.1922])
xlabel(label_index{1,index_max})
ylabel(label_index{1,length(label_index)})
legend('Normal value','data fit line','Anomaly vale','Lower bound','Upper bound')

disp("存在异常值的行为：")
disp(abnorm_index_all1')
index_ab_unique=abnorm_index_all1';


elseif detect_label==6
    [Mdl,tf,scores] = ocsvm(x1, ...
    KernelScale="auto",StandardizeData=true,ContaminationFraction=0.05);
    % [Mdl,tf,scores] = iforest(x,ContaminationFraction=0.05);
    histogram(scores)
    xline(Mdl.ScoreThreshold,"r-",["Threshold" Mdl.ScoreThreshold])
    index_ab_unique=find(tf==1);
    disp("存在异常值的行为：")
    disp(index_ab_unique')
    AA=x1;
    AA(index_ab_unique,:)=[];
    y=AA;
    T = tsne(x, Standardize = true);

    %  绘制可视化结果
    figure
    gscatter(T(:, 1), T(:, 2), tf, "kr", [], 8, "off")
    legend("正常值", "离群值")
    title("One-Class SVM")
    set(gcf,'color','w')
    grid on
    xlabel('dimension1')
    ylabel('dimension2')

elseif detect_label==7
    [Mdl,tf,scores] = rrcforest(x1,ContaminationFraction=0.05,StandardizeData=true);

    histogram(scores)
    xline(Mdl.ScoreThreshold,"r-",["Threshold" Mdl.ScoreThreshold])
    index_ab_unique=find(tf==1);
    disp("存在异常值的行为：")
    disp(index_ab_unique')
    AA=x1;
    AA(index_ab_unique,:)=[];
    y=AA;
    T = tsne(x, Standardize = true);

    %  绘制可视化结果
    figure
    gscatter(T(:, 1), T(:, 2), tf, "kr", [], 8, "off")
    legend("正常值", "离群值")
    title("Robust Random Cut Forest")
    set(gcf,'color','w')
    grid on
    xlabel('dimension1')
    ylabel('dimension2')

elseif detect_label==8
[Mdl,tf,scores] = lof(x1, ...
    ContaminationFraction=0.05, ...
    NumNeighbors=500,Distance="mahalanobis");
    % [Mdl,tf,scores] = lof(x1, ContaminationFraction=0.05,NumNeighbors=500,Distance="mahalanobis");

    histogram(scores)
    xline(Mdl.ScoreThreshold,"r-",["Threshold" Mdl.ScoreThreshold])
    index_ab_unique=find(tf==1);
    disp("存在异常值的行为：")
    disp(index_ab_unique')
    AA=x1;
    AA(index_ab_unique,:)=[];
    y=AA;
    T = tsne(x, Standardize = true);

    %  绘制可视化结果
    figure
    gscatter(T(:, 1), T(:, 2), tf, "kr", [], 8, "off")
    legend("正常值", "离群值")
    title("Local Outlier Factor")
    set(gcf,'color','w')
    grid on
    xlabel('dimension1')
    ylabel('dimension2')

end