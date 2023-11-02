imds = imageDatastore('AT&TData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.Files = natsortfiles(imds.Files);

imdsO = imageDatastore('AT&TData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
sortTest = natsort(imdsO.Labels);
imds.Labels = sortTest;
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7);

net = vgg19;

pixelRange = [-30,30];
inputSize = net.Layers(1).InputSize

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal, 'ColorPreprocessing', 'gray2rgb');

layersTransfer = net.Layers(1:end-3);
numClasses = 40;
layers = [
layersTransfer
fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
softmaxLayer
classificationLayer];

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

% Extract features and find cos similarity between enrollment and verification
layer = "fc7";
featuresTrain = activations(netTransfer,augimdsTrain,layer, OutputAs="rows");
featuresVal = activations(netTransfer,augimdsValidation,layer, OutputAs="rows");

% I think the function I used for cossine similarity may be incorrect.
% For the final project, I will make sure to save the feature matrix
% and write cossine similairty from scratch to see if the getCosineSimilarity()
% calculates it differently.
 
% writematrix(featuresTrain, 'featTrain.csv', 'Delimiter', ',');
% writematrix(featuresVal, 'featVal.csv', 'Delimiter', ',');

% Cosine similarity of the validation set for each subject

valCosEach = double.empty(0, 120);
q = 1;

while (q < 121)
    % Compare f1_8 to fn_9, fn_10, where n = 1:40
    valcose1 = getCosineSimilarity( featuresVal(q,:) , featuresVal(q+1,:));
    valcose2 = getCosineSimilarity( featuresVal(q,:) , featuresVal(q+2,:));

    % Save cosine similarity scores for each enrollment and varification pair
    valCosEach = [valCosEach; valcose1, valcose2];

    % Skip the first image feature of the next subject
    q = q + 3;
end

% Create genuine and imposter score sets for the validation features
valCos = double.empty(0, 120);
i = 2;

while (i < 121)
    % Compare f1_8 to fn_9, fn_10, where n = 1:40
    valcos1 = getCosineSimilarity( featuresVal(1,:) , featuresVal(i,:));
    valcos2 = getCosineSimilarity( featuresVal(1,:) , featuresVal(i+1,:));

    % Save cosine similarity scores for each enrollment and varification pair
    valCos = [valCos; valcos1, valcos2];

    % Skip the first image feature of the next subject
    i = i + 3;
end

testCos = double.empty(0, 280);
j = 2;

while (j < 281)
    % Compare f1_1 to fn_m, where n = 1:40 and m = 2:7
    tecos2 = getCosineSimilarity( featuresTrain(1,:) , featuresTrain(j,:));
    tecos3 = getCosineSimilarity( featuresTrain(1,:) , featuresTrain(j+1,:));
    tecos4 = getCosineSimilarity( featuresTrain(1,:) , featuresTrain(j+2,:));
    tecos5 = getCosineSimilarity( featuresTrain(1,:) , featuresTrain(j+3,:));
    tecos6 = getCosineSimilarity( featuresTrain(1,:) , featuresTrain(j+4,:));
    tecos7 = getCosineSimilarity( featuresTrain(1,:) , featuresTrain(j+5,:));

    % Save cosine similarity scores for each enrollment and varification pair
    testCos = [testCos; tecos2, tecos3, tecos4, tecos5, tecos6, tecos7];
    
    % Skip the first image feature of the next subject
    j = j + 7;
end

writematrix(valCos, 'valCos.csv', 'Delimiter', ',');
writematrix(testCos, 'testCos.csv', 'Delimiter', ',');
writematrix(valCosEach, 'valCosEach.csv', 'Delimiter', ',');

% I NEED TO REMAKE THE HISTOGRAMS AFTER I FIND THE TP AND FP SCORES
% BASED ON HOW CLOSE COSSIM IS TO 1

% Obtain plots and values for training set
% Rename columns based on which verification image was compaired to the 
% enrollment image. 
% ie. Verification Image1 = fn_2, Verification Image2 = fn_3, etc.
allVars = 1:width(testCos);
newNames = append("Verification Image",string(allVars));
testCos = renamevars(testCos,allVars,newNames);

% Initialize arrays to store TPR and FPR values for each verification image.
tpr_allT = zeros(numel(newNames), numel(thresholds));
fpr_allT = zeros(numel(newNames), numel(thresholds));

tp_allT = zeros(numel(newNames), numel(thresholds));
fp_allT = zeros(numel(newNames), numel(thresholds));

% Assign labels to the cosine similarity output where the only true value
% are the first two entries per column. This is the case because the identity image
% belongs to class number one and only the first two entries per each column
% are from the class number one.
labels = zeros(1, 40);
labels(1:1) = 1;
labels = labels';

% Calculate tp and fp for each cosine similarity based on how close they are to a
% range of values between 0 and 1.
thresholds = linspace(0, 1, 50);

% Create a single figure for all 6 verificaion ROC curves and AUC scores.
figure;

% Loop through each column
for j = 1:numel(newNames)
    cosineSimTest = testCos.(newNames{j});
    
    % Initialize arrays to store TPR and FPR values for the current column
    tprT = zeros(1, numel(thresholds));
    fprT = zeros(1, numel(thresholds));

    tpTs = zeros(1, numel(thresholds));
    fpTs = zeros(1, numel(thresholds));

    % Loop through each threshold
    for i = 1:numel(thresholds)
        threshold = thresholds(i);
        
        % Predict labels based on proximity to the current threshold
        predicted_labelsT = double(cosineSimTest >= threshold);

        % Calculate tp and fp
        fpT = sum(labels == 0 & predicted_labelsT == 1);
        fpTs(i) = fpT;
        tnT = sum(labels == 0 & predicted_labelsT == 0);
        
        fnT = sum(labels == 1 & predicted_labelsT == 0);
        tpT = sum(labels == 1 & predicted_labelsT == 1);
        tpTs(i) = tpT;

        % Calculate TPR and FPR
        tprT(i) = tpT / (tpT + fnT);
        fprT(i) = fpT / (tnT + fpT);

    end

    % Store TPR and FPR values for the current column
    tpr_allT(j, :) = tprT;
    fpr_allT(j, :) = fprT;

    % Store genuine (tp) and imposter (fp) scores per threshold
    tp_allT(j, :) = tpTs;
    fp_allT(j, :) = fpTs;
    
    % Plot ROC curve for the current column on the same figure
    hold on;
    plot(fprT, tprT);
    
    % Label the ROC curve
    legend_labels{j} = ['ROC Curve for ', newNames{j}];
end

% Set axis labels, title, and legend
xlabel('Test Set False Positive Rate (FPR)');
ylabel('Test Set True Positive Rate (TPR)');
title('ROC Curves for Each Train Verification Image');
grid on;

% Add a legend for each ROC curve
legend(legend_labels, 'Location', 'Best');

% tp histogram
tp_allT = tp_allT';
fp_allT = fp_allT';

posNegNameT = ["True Positive" "False Positive" "True Positive" "False Positive" "True Positive" "False Positive"];

for j = 1:width(tp_allT)
    figure;
    hold on;
    histogram(tp_allT(:,j));
    histogram(fp_allT(:,j));
    title(['Histogram for Training', newNames{j}]);

    % Label the bins
    legend_labels{j} = ['Bins for ', posNegNameT{j}];

    % Set axis labels, title, and legend
    xlabel('Threshold Value');
    ylabel('Number of tp (blue) or fp (orange)');
    grid on;

    % Add a legend for tp and fp
    legend(legend_labels, 'Location', 'Best');
    hold off;
end

% Calculate AUC (Area Under the Curve) for each column
for j = 1:numel(newNames)
    roc_aucT = trapz(fpr_allT(j, :), tpr_allT(j, :));
    disp(['AUC for ', newNames{j}, ': ', num2str(roc_aucT)]);
end

% Obtain plots and values for validation set
% Rename columns based on which verification image was compaired to the 
% enrollment image. 
% ie. Verification Image1 = fn_2, Verification Image2 = fn_3, etc.
allVarsV = 1:width(valCos);
newNamesV = append("Verification Image",string(allVarsV));
valCos = renamevars(valCos,allVarsV,newNamesV);

% Calculate tp and fp for each cosine similarity based on how close they are to a
% range of values between 0 and 1.
thresholds = linspace(0, 1, 50);

% Initialize arrays to store TPR and FPR values for each verification image.
tpr_allV = zeros(numel(newNamesV), numel(thresholds));
fpr_allV = zeros(numel(newNamesV), numel(thresholds));

tp_allV = zeros(numel(newNamesV), numel(thresholds));
fp_allV = zeros(numel(newNamesV), numel(thresholds));

% Assign labels to the cosine similarity output where the only true value
% are the first two entries per column. This is the case because the identity image
% belongs to class number one and only the first two entries per each column
% are from the class number one.
labels = zeros(1, 40);
labels(1:1) = 1;
labels = labels';

% Create a single figure for all 6 verificaion ROC curves and AUC scores.
figure;

% Loop through each column
for j = 1:numel(newNamesV)
    cosineSimVal = valCos.(newNamesV{j});
    
    % Initialize arrays to store TPR and FPR values for the current column
    tprV = zeros(1, numel(thresholds));
    fprV = zeros(1, numel(thresholds));

    tpVs = zeros(1, numel(thresholds));
    fpVs = zeros(1, numel(thresholds));
    
    % Loop through each threshold
    for i = 1:numel(thresholds)
        threshold = thresholds(i);
        
        % Predict labels based on proximity to the current threshold
        predicted_labelsV = double(cosineSimVal >= threshold);

        % Calculate tp and fp
        fpV = sum(labels == 0 & predicted_labelsV == 1);
        fpVs(i) = fpV;
        tnV = sum(labels == 0 & predicted_labelsV == 0);
        
        fnV = sum(labels == 1 & predicted_labelsV == 0);
        tpV = sum(labels == 1 & predicted_labelsV == 1);
        tpVs(i) = tpV;

        % Calculate TPR and FPR
        tprV(i) = tpV / (tpV + fnV);
        fprV(i) = fpV / (tnV + fpV);
    end

    % Store TPR and FPR values for the current column
    tpr_allV(j, :) = tprV;
    fpr_allV(j, :) = fprV;

    tp_allV(j, :) = tpVs;
    fp_allV(j, :) = fpVs;
    
    % Plot ROC curve for the current column on the same figure
    hold on;
    plot(fprV, tprV);
    
    % Label the ROC curve
    legend_labels{j} = ['ROC Curve for ', newNamesV{j}];

end

% Set axis labels, title, and legend
xlabel('Val Set False Positive Rate (FPR)');
ylabel('Val Set True Positive Rate (TPR)');
title('ROC Curves for Each Validation Verification Image');
grid on;

% Add a legend for each ROC curve
legend(legend_labels, 'Location', 'Best');

% tp histogram
tp_allV = tp_allV';
fp_allV = fp_allV';

posNegName = ["True Positive" "False Positive"];

for j = 1:width(tp_allV)
    figure;
    hold on;
    histogram(tp_allV(:,j));
    histogram(fp_allV(:,j));
    title(['Histogram for Validation', newNamesV{j}]);

    % Label the bins
    legend_labels{j} = ['Bins for ', posNegName{j}];

    % Set axis labels, title, and legend
    xlabel('Threshold Value');
    ylabel('Number of tp (blue) or fp (orange)');
    grid on;

    % Add a legend for tp and fp
    legend(legend_labels, 'Location', 'Best');
    hold off;
end

% Calculate AUC (Area Under the Curve) for each column
for j = 1:numel(newNamesV)
    roc_aucV = trapz(fpr_allV(j, :), tpr_allV(j, :));
    disp(['AUC for ', newNamesV{j}, ': ', num2str(roc_aucV)]);
end

