% Leif Wesche
% MNIST Machine Learning and Classification

%% Load Images
close all
clear all
clc

test_dat=csvread('test.csv');
train_dat=csvread('train.csv');
res=[28, 28];

%% View Training Images
% close all
% clc
% 
% i=18545;
% train(i, 1)
% im=train(i,2:end);
% im=reshape(im, res);
% pcolor(im), colormap(gray), shading interp
% 

%%

close all
clc


for i=[1:length(train_dat)]
dum=train_dat(i, 2:end);
dum=reshape(dum, 28, 28);
train(:,:,1,i)=dum;
end    

for i=[1:length(test_dat)]
dum=test_dat(i, :);    
dum=reshape(dum, 28, 28);
Xtest(:,:,1,i)=dum;
end

label=train_dat(:,1);
label=categorical(label);

q=randperm(length(train_dat));

Xtrain=train(:,:,:,(q(1:38000)));
Xtrain_label=label(q(1:38000));
Xvalidate=train(:,:,:,(q(38001:end)));
Xvalidate_label=label(q(38001:end));
size(Xtrain);
size(Xvalidate);


%% Test1 Validation
close all
clc

accuracy1=zeros(1, 5);
for j=[1:5]
    
layers=[imageInputLayer([res, 1]);
        convolution2dLayer(5,20);
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(Xtrain, Xtrain_label, layers, options);

Yvalidate=classify(net, Xvalidate);

dif=double(Yvalidate)-double(Xvalidate_label);

dum=0;
for i=[1:length(dif)] 
if dif(i)~=0;
    dum=dum+1;
end 
end
accuracy1(j)=(1-dum/length(dif))*100;

end



%% Test2 Validation
close all
clc

accuracy2=zeros(1, 5);
for j=[1:5]
    
layers=[imageInputLayer([res, 1]);
        maxPooling2dLayer(2,'Stride',2);
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(Xtrain, Xtrain_label, layers, options);

Yvalidate=classify(net, Xvalidate);

dif=double(Yvalidate)-double(Xvalidate_label);

dum=0;
for i=[1:length(dif)] 
if dif(i)~=0;
    dum=dum+1;
end 
end
accuracy2(j)=(1-dum/length(dif))*100;

end



%% Test3 Validation
close all
clc

accuracy3=zeros(1, 5);
for j=[1:5]
    
layers=[imageInputLayer([res, 1]);
        reluLayer();
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(Xtrain, Xtrain_label, layers, options);

Yvalidate=classify(net, Xvalidate);

dif=double(Yvalidate)-double(Xvalidate_label);

dum=0;
for i=[1:length(dif)] 
if dif(i)~=0;
    dum=dum+1;
end 
end
accuracy3(j)=(1-dum/length(dif))*100;

end



%% Test4 Validation
close all
clc

accuracy4=zeros(1, 5);
for j=[1:5]
    
layers=[imageInputLayer([res, 1]);
        convolution2dLayer(10,40);
        reluLayer();
        maxPooling2dLayer(2,'Stride',2);
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(Xtrain, Xtrain_label, layers, options);

Yvalidate=classify(net, Xvalidate);

dif=double(Yvalidate)-double(Xvalidate_label);

dum=0;
for i=[1:length(dif)] 
if dif(i)~=0;
    dum=dum+1;
end 
end
accuracy4(j)=(1-dum/length(dif))*100;

end




%% Plot data
close all
clc

load('T1_accuracy')
load('T2accuracy')
load('T3_accuracy')
load('T4_accuracy')

figure
subplot(2,2,1)
bar(accuracy2, 'g')
title('A) Network 1 Verification'); xlabel('Trial #'); ylabel('Accuracy (%)'); grid on

subplot(2,2,2)
bar(accuracy3, 'r')
title('B) Network 2 Verification'); xlabel('Trial #'); ylabel('Accuracy (%)'); grid on

subplot(2,2,3)
bar(accuracy4, 'y')
title('C) Network 3 Verification'); xlabel('Trial #'); ylabel('Accuracy (%)'); grid on

subplot(2,2,4)
bar(accuracy, 'b')
title('D) Network 4 Verification'); xlabel('Trial #'); ylabel('Accuracy (%)'); grid on


%% Test1
close all
clc
    
layers=[imageInputLayer([res, 1]);
        convolution2dLayer(5,20);
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(train, label, layers, options);

ytest=classify(net, Xtest);

ytest=double(ytest)-1;

output=[];
output=[(1:28000)', ytest];
headers={'ImageId', 'Label'};

csvwrite_with_headers('test1', output, headers);



%% Test2
close all
clc
    
layers=[imageInputLayer([res, 1]);
        maxPooling2dLayer(2,'Stride',2);
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(train, label, layers, options);

ytest=classify(net, Xtest);

ytest=double(ytest)-1;

output=[];
output=[(1:28000)', ytest];
headers={'ImageId', 'Label'};

csvwrite_with_headers('test2', output, headers);


%% Test3
close all
clc
    
layers=[imageInputLayer([res, 1]);
        reluLayer();
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(train, label, layers, options);

ytest=classify(net, Xtest);

ytest=double(ytest)-1;

output=[];
output=[(1:28000)', ytest];
headers={'ImageId', 'Label'};

csvwrite_with_headers('test3', output, headers);



%% Test4
close all
clc
    
layers=[imageInputLayer([res, 1]);
        convolution2dLayer(10,40);
        reluLayer();
        maxPooling2dLayer(2,'Stride',2);
        fullyConnectedLayer(10);
        softmaxLayer();
        classificationLayer()];
   
options=trainingOptions('sgdm', 'MaxEpochs', 20, 'InitialLearnRate', 0.0001);

net=trainNetwork(train, label, layers, options);

ytest=classify(net, Xtest);

ytest=double(ytest)-1;

output=[];
output=[(1:28000)', ytest];
headers={'ImageId', 'Label'};

csvwrite_with_headers('test4', output, headers);


