unzip('GTdb_crop.zip');

imds = imageDatastore('cropped_faces', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Need to get labels for each file and label each file s01 to s50
[path, name] =  fileparts(imds.Files);

% Extract labels from the file names
labelsCellArray = cellfun(@(x) x(1:3), name, 'UniformOutput', false);

% Convert the cell array of labels to a categorical array
labelsCat = categorical(labelsCellArray);

% Assign labels to each file in imds.
imds.Labels = labelsCat;

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.33, 0.33, 'randomized');

net = alexnet;

pixelRange = [-30,30];
inputSize = net.Layers(1).InputSize;

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    					'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal);

augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

layers = [
	layersTransfer
	fullyConnectedLayer(numClasses,'WeightLearnRateFactor',25,'BiasLearnRateFactor',25)
	softmaxLayer
	classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',5, ...  
    'MaxEpochs',7, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment', 'cpu',...  
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
	'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

save netTransfer 'netTransfer'

% Measure goodness based on it's classification accuracy
[YPred,scores] = classify(netTransfer,augimdsValidation);

YValidation = imdsVal.Labels;
accuracyV = mean(YPred == YValidation) % 74.8

[YPredT,scoresT] = classify(netTransfer,augimdsTest);

YTest = imdsTest.Labels;
accuracyT = mean(YPredT == YTest) % 74.8

%%% Extract features and find cos similarity between enrollment and verification

layer = "fc7";
featuresTest = activations(netTransfer, augimdsTest, layer, OutputAs="rows");

% Save features to avoid having to train the network multiple times.

writematrix(featuresTest, 'featTest.csv', 'Delimiter', ',');

% Create genuine and imposter score sets for the validation features

testCos = double.empty(0, 50);
j = 2;

while (j < 251)
    % Compare f1_1 to fn_m, where n = 1:50 and m is the 2nd, 3rd, 4th, and 5th image per input class
    tecos1 = getCosineSimilarity( featuresTest(1,:) , featuresTest(j,:));
    tecos2 = getCosineSimilarity( featuresTest(1,:) , featuresTest(j+1,:));
    tecos3 = getCosineSimilarity( featuresTest(1,:) , featuresTest(j+2,:));
    tecos4 = getCosineSimilarity( featuresTest(1,:) , featuresTest(j+3,:));

    % Save cosine similarity scores for each enrollment and varification pair
    testCos = [testCos; tecos1, tecos2, tecos3, tecos4];
    
    % Skip the first image feature of the next subject
    j = j + 5;
end

writematrix(testCos, 'testCos.csv', 'Delimiter', ',');

%%% AUC, ROC, and histogram of combining cosine similarity scores into one column. I will use the threshold of this to find 
%%% rank 1 and rank 5 scores.

% Append testCosArray from size 4 columns and 50 rows to 1 column 200 rows
testCosArrayAll = reshape(testCos, [200, 1]);

% Every 50th score is comparing two images of the same person, therefore a true match. 
labelsAll = zeros(200, 1);
labelsAll(50:50:end) = 1;

%% ROC and AUC
[XAll,YAll,TAll,AUCAll,OPTROCPTAll] = perfcurve(labelsAll,testCosArrayAll,1);

% Save output of perfcurve for future use in rank 1 and rank 5 scores
perFcurvOut = {XAll,YAll,TAll,AUCAll,OPTROCPTAll};
save('AUC_ROC_Out_FinalProject', 'perFcurvOut');

% Plot ROC with AUC score
figure;

hold on;
plot(XAll,YAll);

% Set axis labels, title, and legend
xlabel('Test Set False Positive Rate (FPR)');
ylabel('Test Set True Positive Rate (TPR)');
title('ROC Curve for all with AUC of ', num2str(AUCAll));
grid on;

hold off;

%% Histogram

% Variables for use in Histogram and d' calculation
tpXAll = XAll;
fnXAll = 1 - XAll;
fpYAll = YAll;

figure;
hold on;
histogram(tpXAll);
histogram(fpYAll);
title(['Histogram for Testing with All Scores']);

% Set axis labels, title, and legend
xlabel('Threshold Value');
ylabel('Number of tp (blue) or fp (orange)');
grid on;

hold off;

%% Find d' for testing set.

muTPAll = mean(tpXAll);
muFNAll = mean(fnXAll);
StdvTAll = std(tpXAll);
StdvPAll = std(fnXAll);

dPrimeAll = abs((muTPAll - muFNAll)) / StdvTAll;

disp(['The d' value is ', num2string(dPrimeAll)]); % 0.012483

%%% rank 1 and rank 5

% Find cosSim using first image per subject in the test subset. This will result in 50 sets of 200 cosine similarity scores.

% Array that will aid in cycling through every 5th image as the enrollment images for cosine similarity.
rankSteps = 1:5:250;
rankSteps = rankSteps';

% Empty cell to store cossine similarity scores per enrollment image. 50 total.
cosTestCells = {};

for k = 1:50
	
	enrol = rankSteps(k);
	testCosR = double.empty(0, 50);
	j = 1;

	while (j < 251)
    	% Compare f1_1 to fn_m, where n = 1:50 and m = 2:5
    	tecos1 = getCosineSimilarity( featuresTest(enrol,:) , featuresTest(j+1,:));
    	tecos2 = getCosineSimilarity( featuresTest(enrol,:) , featuresTest(j+2,:));
    	tecos3 = getCosineSimilarity( featuresTest(enrol,:) , featuresTest(j+3,:));
    	tecos4 = getCosineSimilarity( featuresTest(enrol,:) , featuresTest(j+4,:));

    	% Save cosine similarity scores for each enrollment and varification pair
    	testCosR = [testCosR; tecos1, tecos2, tecos3, tecos4];
    
    	% Skip the first image feature of the next subject
    	j = j + 5;
	end

	% Put scores in order for rank analysis
	testCosRAll = reshape(testCosR, [200, 1]);
	testCosRSort = sort(testCosRAll,'descend');

	cosTestCells{k} = testCosRSort;
end 

% Save same sample image comparisons where j is the verification image and j+n are the four images of the same subject

testCosImage = double.empty(0, 50);
j = 1;

while (j < 250)
	
    % Compare f1_1 to fn_m, where n = 1:50 and m = 2:5
    tecos1 = getCosineSimilarity( featuresTest(j,:) , featuresTest(j+1,:));
   	tecos2 = getCosineSimilarity( featuresTest(j,:) , featuresTest(j+2,:));
   	tecos3 = getCosineSimilarity( featuresTest(j,:) , featuresTest(j+3,:));
   	tecos4 = getCosineSimilarity( featuresTest(j,:) , featuresTest(j+4,:));
	
    % Save cosine similarity scores for each enrollment and verification pair
    testCosImage = [testCosImage; tecos1, tecos2, tecos3, tecos4];

	j = j + 5

end 

% Save cosine similarity scores comparing first image of each subject to itself 'testCosImage.csv' 
% and scores of comparing first image of each sample to the second through fifth images of all subjects 'cosTestCells.mat'
writematrix(testCosImage, 'testCosImage.csv', 'Delimiter', ',');
save('cosTestCells.mat', 'cosTestCells');

%% Rank 1 calculation

% Put scores in order for rank analysis
testCosRank = reshape(testCosImage, [200, 1]);
testCosRank = sort(testCosRank,'descend');

% Extract the top score for the case where the enrollment image is from the first sample
maxVal = max(testCosRank);
maxVal = round(maxVal, 4);

% Find optimal single value where true positive equals false positive. Since the output of OPTROCPT are two values, I opted
% to find the point across the curve where true positives equal false positives.
ThresholdForOptROCpt = TAll((XAll==OPTROCPTAll(1))&(YAll==OPTROCPTAll(2)));
ThresholdForOptROCpt = round(ThresholdForOptROCpt, 4);

% Round values in testCosImage to four decimal places
testCosImage = round(testCosImage, 4);

% Declare empty scalar for counting number of successes
successRank1 = 0;
i = 1;

% Compare maxVal to optimal threshold and see if the max value of the first subject matches any of the cosine similarity 
% scores of the images in that same sample
for i = 1:50
	if maxVal >= ThresholdForOptROCpt & any(ismember(maxVal, testCosImage(i, :)))

		successRank1 = successRank1 + 1;

	else

		continue;

	end
end

% Take average of all successes.
averageRank1 = successRank1/50; % = 0.0200

% Find the top 5 scores from the sorted list: load 'cosTestCells.mat' and make a 1 row 4 column matrix with the top four 
% scores

% Extract the top score for the case where the enrollment image is from the first sample
topFive = testCosRank(1:5);
topFive = round(topFive, 4);

successRank5 = 0;

for i = 1:50
	if any(topFive >= ThresholdForOptROCpt) & any(ismember(topFive, testCosImage(i, :)))

		successRank5 = successRank5 + 1;

	else

		continue;
	end

end

% Take average of all successes.
averageRank5 = successRank5/50; % = 0.04
