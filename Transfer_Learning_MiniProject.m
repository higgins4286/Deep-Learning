%%% Lauren Higgins MiniProject 4 Code

%%% Part A: AlexNet 

unzip(AT&T_Data.zip);
imds = imageDatastore('AT&TData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
for i = 1:16
subplot(4,4,i)
I = readimage(imdsTrain, idx(i));
imshow(I)
end 

net = alexnet; 

analyzeNetwork(net)
layersTransfer = net.Layers(1:end-3);
numClasses = 40

numClasses =

    40

layers = [
layersTransfer
fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
softmaxLayer
classificationLayer];

pixelRange = [-30,30];

inputSize = net.Layers(1).InputSize

inputSize =

   227   227     3

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% Added 'ColorPreprocessing', 'gray2rgb' to correct input size error.
% netTransfer = trainNetwork(augimdsTrain,layers,options);
% Error using trainNetwork (line 184)
% The training images are of size 227×227×1 but the input layer expects images of size 227×227×3.
 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal, 'ColorPreprocessing', 'gray2rgb');

% Added an extra parameter to trainingOptions to correct following error
% netTransfer = trainNetwork(augimdsTrain,layers,options);
% Error using trainNetwork (line 184)
% Unexpected error calling cuDNN: CUDNN_STATUS_BAD_PARAM.

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment', 'cpu',...  % Added to fix above cuDNN error
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);
 
[YPred,scores] = classify(netTransfer,augimdsValidation);
idx = randperm(numel(imdsVal.Files),4);

figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsVal,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YValidation = imdsVal.Labels;
accuracy = mean(YPred == YValidation)

accuracy =

    0.7500

%%%  Part B: VGG 19

imds = imageDatastore('AT&TData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
net = vgg19;
analyzeNetwork(net)

layersTransfer = net.Layers(1:end-3);
numClasses = 40;

layers = [
	layersTransfer
	fullyConnectedLayer(numClasses,
		             'WeightLearnRateFactor', 20,
		             'BiasLearnRateFactor', 20)
	softmaxLayer
classificationLayer];

pixelRange = [-30,30];
inputSize = net.Layers(1).InputSize

inputSize =

   224   224     3

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal, 'ColorPreprocessing', 'gray2rgb');

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment', 'cpu',...  
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);
[YPred,scores] = classify(netTransfer, augimdsValidation);
idx = randperm(numel(imdsVal.Files),4);

figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsVal,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
YValidation = imdsVal.Labels;
accuracy = mean(YPred == YValidation)

accuracy =

    0.6750
