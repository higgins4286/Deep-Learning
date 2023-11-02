%%% Part A

% Run original code

if ~isfile('PhysionetData.mat')
    ReadPhysionetData         
end
load PhysionetData

Signals(1:5)

ans =

  5×1 cell array

    {1×9000  double}
    {1×9000  double}
    {1×18000 double}
    {1×9000  double}
    {1×18000 double}

Labels(1:5)

ans = 

  5×1 categorical array

     N 
     N 
     N 
     A 
     A 

summary(Labels)
     A       738 
     N      5050 

L = cellfun(@length,Signals);
h = histogram(L);
xticks(0:3000:18000);
xticklabels(0:3000:18000);
title('Signal Lengths')
xlabel('Length')
ylabel('Count')

normal = Signals{1};
aFib = Signals{4};

subplot(2,1,1)
plot(normal)
title('Normal Rhythm')
xlim([4000,5200])
ylabel('Amplitude (mV)')
text(4330,150,'P','HorizontalAlignment','center')
text(4370,850,'QRS','HorizontalAlignment','center')

subplot(2,1,2)
plot(aFib)
title('Atrial Fibrillation')
xlim([4000,5200])
xlabel('Samples')
ylabel('Amplitude (mV)')

[Signals,Labels] = segmentSignals(Signals,Labels);
Signals(1:5)

ans =

  5×1 cell array

    {1×9000 double}
    {1×9000 double}
    {1×9000 double}
    {1×9000 double}
    {1×9000 double}

summary(Labels)
A       718 
N      4937

afibX = Signals(Labels=='A');
afibY = Labels(Labels=='A');

normalX = Signals(Labels=='N');
normalY = Labels(Labels=='N');

[trainIndA,~,testIndA] = dividerand(718,0.9,0.0,0.1);
[trainIndN,~,testIndN] = dividerand(4937,0.9,0.0,0.1);

XTrainA = afibX(trainIndA);
YTrainA = afibY(trainIndA);

XTrainN = normalX(trainIndN);
YTrainN = normalY(trainIndN);

XTestA = afibX(testIndA);
YTestA = afibY(testIndA);

XTestN = normalX(testIndN);
YTestN = normalY(testIndN);

XTrain = [repmat(XTrainA(1:634),7,1); XTrainN(1:4438)];
YTrain = [repmat(YTrainA(1:634),7,1); YTrainN(1:4438)];

XTest = [repmat(XTestA(1:70),7,1); XTestN(1:490)];
YTest = [repmat(YTestA(1:70),7,1); YTestN(1:490);];

summary(YTrain)
     A      4438 
     N      4438 
summary(YTest)
     A      490 
     N      490

layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]

layers = 

  5×1 Layer array with layers:

     1   ''   Sequence Input          Sequence input with 1 dimensions
     2   ''   BiLSTM                  BiLSTM with 100 hidden units
     3   ''   Fully Connected         2 fully connected layer
     4   ''   Softmax                 softmax
     5   ''   Classification Output   crossentropyex

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);
trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy = sum(trainPred == YTrain)/numel(YTrain)*100

LSTMAccuracy =

   60.4439   % Training Accuracy

% LSTMAccuracy is slightly different than example due to randimization of
% [trainIndA,~,testIndA] = dividerand(718,0.9,0.0,0.1);
% [trainIndN,~,testIndN] = dividerand(4937,0.9,0.0,0.1);

figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');

testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracyTest = sum(testPred == YTest)/numel(YTest)*100

LSTMAccuracyTest =

   55.6122

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');

% Better version with a bilstm of size 300 with the same initial 
% training options.

layers2 = [ ...
    sequenceInputLayer(1)
    bilstmLayer(300,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]

 5×1 Layer array with layers:

     1   ''   Sequence Input          Sequence input with 1 dimensions
     2   ''   BiLSTM                  BiLSTM with 300 hidden units
     3   ''   Fully Connected         2 fully connected layer
     4   ''   Softmax                 softmax
     5   ''   Classification Output   crossentropyex

options2 = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net3 = trainNetwork(XTrain,YTrain,layers2,options2);
trainPred2 = classify(net3,XTrain,'SequenceLength',1000);
LSTMAccuracy2 = sum(trainPred2 == YTrain)/numel(YTrain)*100

LSTMAccuracy2 =

   60.6580

testPred = classify(net3,XTest,'SequenceLength',1000);
LSTMAccuracyTest2 = sum(testPred == YTest)/numel(YTest)*100

LSTMAccuracyTest2 =

   59.5918

figure
confusionchart(YTrain,trainPred2,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM Train 300 Layers');

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM Test 300 Layers');

% Better version with a bilstm of size 300 with an InitialLearnRate of 
% 0.001.

options3 = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.001, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net4 = trainNetwork(XTrain,YTrain,layers2,options3);
trainPred3 = classify(net4,XTrain,'SequenceLength',1000);
LSTMAccuracy3 = sum(trainPred3 == YTrain)/numel(YTrain)*100

LSTMAccuracy3 =

   85.4552

testPred3 = classify(net4,XTest,'SequenceLength',1000);
LSTMAccuracyTest3 = sum(testPred3 == YTest)/numel(YTest)*100

LSTMAccuracyTest3 =

   84.6939

figure
confusionchart(YTrain,trainPred3,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM Train 300 Layers LR 0.001');
figure
confusionchart(YTest,testPred3,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM Test 300 Layers LR 0.001');

%% Bonus regular LSTM 

% Initial set up as in the example.
layersReg = [ ...
    sequenceInputLayer(1)
    lstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

netReg = trainNetwork(XTrain,YTrain,layersReg,options);

trainPredReg = classify(netReg,XTrain,'SequenceLength',1000);
LSTMAccuracyReg = sum(trainPredReg == YTrain)/numel(YTrain)*100

LSTMAccuracyReg =

   56.1176

testPredReg = classify(netReg,XTest,'SequenceLength',1000);
LSTMAccuracyTestReg = sum(testPredReg == YTest)/numel(YTest)*100

LSTMAccuracyTestReg =

   55.5102

figure
confusionchart(YTrain,trainPredReg,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Train Initial Layers');
figure
confusionchart(YTest,testPredReg,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Test Initial Layers');

% Applying the same paramaters of my first improved model.

layersReg2 = [ ...
    sequenceInputLayer(1)
    lstmLayer(300,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]

netReg2 = trainNetwork(XTrain,YTrain,layersReg2,options);
trainPredReg2 = classify(netReg2,XTrain,'SequenceLength',1000);
LSTMAccuracyReg2 = sum(trainPredReg2 == YTrain)/numel(YTrain)*100

LSTMAccuracyReg2 =

   55.7571

testPredReg2 = classify(netReg2,XTest,'SequenceLength',1000);
LSTMAccuracyTestReg2 = sum(testPredReg2 == YTest)/numel(YTest)*100

LSTMAccuracyTestReg2 =

   53.2653

figure
confusionchart(YTrain,trainPredReg2,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Train 300 Layers');
figure
confusionchart(YTest,testPredReg2,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Test 300 Layers');

% Apply second improved network from previous example.

optionsReg = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.001, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

netReg3 = trainNetwork(XTrain,YTrain,layersReg,optionsReg);
trainPredReg3 = classify(netReg3,XTrain,'SequenceLength',1000);
LSTMAccuracyReg3 = sum(trainPredReg3 == YTrain)/numel(YTrain)*100

LSTMAccuracyReg3 =

   87.4493

testPredReg3 = classify(netReg3,XTest,'SequenceLength',1000);
LSTMAccuracyTestReg3 = sum(testPredReg3 == YTest)/numel(YTest)*100

LSTMAccuracyTestReg3 =

   87.6531

figure
confusionchart(YTrain,trainPredReg3,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Train 300 Layers LR 0.001');
figure
confusionchart(YTest,testPredReg3,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Test 300 Layers LR 0.001');

% One more version since the single lstm with size of 200 did worse than 
% the initial size of 100. I will try to decrease the initial size to 50
% and use the 0.001 learning rate.

layersReg50 = [ ...
    sequenceInputLayer(1)
    lstmLayer(50,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ]

netReg4 = trainNetwork(XTrain,YTrain,layersReg50,optionsReg);

trainPredReg4 = classify(netReg4,XTrain,'SequenceLength',1000);
LSTMAccuracyReg4 = sum(trainPredReg4 == YTrain)/numel(YTrain)*100

LSTMAccuracyReg4 =

   81.5119

testPredReg4 = classify(netReg4,XTest,'SequenceLength',1000);
LSTMAccuracyTestReg4 = sum(testPredReg4 == YTest)/numel(YTest)*100

LSTMAccuracyTestReg4 =

   80.3061

figure
confusionchart(YTrain,trainPredReg4,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Train 50 Layers LR 0.001');
figure
confusionchart(YTest,testPredReg4,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for RegLSTM Test 50 Layers LR 0.001');
% Performed worse than the 300 size with same learning rate. Ideal value might be 100 or somewhere between 100 and 300.

%%% Part B

load('PhysionetData.mat')

NewSig = cell(numel(Signals),1);

for n=1:length(Signals)
    
	noisyECG_withTrend=(Signals{n})';   

	[p,s,mu] = polyfit((1:numel(noisyECG_withTrend))',noisyECG_withTrend,6);
	f_y = polyval(p,(1:numel(noisyECG_withTrend))',[],mu);

	ECG_data = noisyECG_withTrend - f_y;        % Detrend data

	[~,locs_Rwave] = findpeaks(ECG_data,'MinPeakHeight',0.5,...
                                    'MinPeakDistance',200);

	NewSigs=diff(locs_Rwave);   

	NewSig{n}=NewSigs(3:end-1)'; %Use this instead as input to LSTMs
end

afibX = NewSig(Labels=='A');
afibY = Labels(Labels=='A');

normalX = NewSig(Labels=='N');
normalY = Labels(Labels=='N');

[trainIndA,~,testIndA] = dividerand(718,0.9,0.0,0.1);
[trainIndN,~,testIndN] = dividerand(4937,0.9,0.0,0.1);

XTrainA = afibX(trainIndA);
YTrainA = afibY(trainIndA);

XTrainN = normalX(trainIndN);
YTrainN = normalY(trainIndN);

XTestA = afibX(testIndA);
YTestA = afibY(testIndA);

XTestN = normalX(testIndN);
YTestN = normalY(testIndN);

XTrain = [repmat(XTrainA(1:634),7,1); XTrainN(1:4438)];
YTrain = [repmat(YTrainA(1:634),7,1); YTrainN(1:4438)];

XTest = [repmat(XTestA(1:70),7,1); XTestN(1:490)];
YTest = [repmat(YTestA(1:70),7,1); YTestN(1:490);];

layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);
net = trainNetwork(XTrain,YTrain,layers,options);

trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy = sum(trainPred == YTrain)/numel(YTrain)*100

LSTMAccuracy =

   50

figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart Train for R-R LSTM');

testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracyTest = sum(testPred == YTest)/numel(YTest)*100

LSTMAccuracyTest =

   50

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart Test for R-R LSTM');

% Trying a better version

layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(300,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.001, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy = sum(trainPred == YTrain)/numel(YTrain)*100

LSTMAccuracy =

   84.3961

figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart Train 300 Layers 0.001 lr for R-R LSTM');

testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracyTest = sum(testPred == YTest)/numel(YTest)*100

LSTMAccuracyTest =

   84.3878

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart Test 300 Layers 0.001 lr for R-R LSTM');

% Second try at a better result with a smaller mini batch size of 100.

layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(300,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 100, ... 
    'InitialLearnRate', 0.001, ...
    'SequenceLength', 1000, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy = sum(trainPred == YTrain)/numel(YTrain)*100

LSTMAccuracy =

   85.3425

figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title',...
              '                             Confusion Chart Train \newline 300 Layers, 0.001 lr, 100 minibatch samples for R-R LSTM');

testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracyTest = sum(testPred == YTest)/numel(YTest)*100

LSTMAccuracyTest =

   86.7347

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title',...
              '                             Confusion Chart Test \newline 300 Layers, 0.001 lr, 100 minibatch samples for R-R LSTM');
