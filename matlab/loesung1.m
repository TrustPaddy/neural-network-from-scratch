clc; clear; close all;

number_of_epochs = 10^5;

q1 = myRandVector(100, -1, 1);
q2 = myRandVector(100, -1, 1);
q = [q1, q2];

x1Label = (q1 + q2 - 1);
x2Label = (q1 - q2 + 1);
xLabel = [x1Label, x2Label];

global arch;
arch = [2, 2, 2, 2];  

w_size = sum(arch(1:end-1) .* arch(2:end)) + sum(arch) - arch(1);

w = myRandVector(w_size, -1, 1)';

steps = 1:number_of_epochs;
losses = zeros(1, number_of_epochs);

%learningRate = 0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NEW %%%%%%%%%%%%%%%%%%%%
alpha = 0.001; % = r
mBeta_1 = 0.9;    % decay rate w.r.t first moment
mBeta_2 = 0.999;  % decay rate w.r.t second moment 
epsilon = 1e-8; % against zero division
weight_decay = 0.0; % <<< ADAMW-spefific
mk = zeros(size(w)); % first moment estimate
v_k = zeros(size(w)); % second moment estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:number_of_epochs

    [x1, x2] = forward(q, w);

    [grad, returnedLoss] = calcGradientOfLossFunction(q, xLabel, w);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NEW%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % w.r.t first
    mk = mBeta_1 * mk + (1 - mBeta_1)*grad;

    % w.r.t second
    v_k = mBeta_2 * v_k + (1 - mBeta_2)*(grad.^2);

    % corrective adaptation
    mHat_k = mk/(1 - mBeta_1^i);
    vHat_k = v_k/(1 - mBeta_2^i);

    % adaptive learning rate
    rk = alpha./ (sqrt(vHat_k) + epsilon);

    % weight update
    %w = w - rk.* mHat_k; %instead of wk = wk - r*grad;
    w = w - rk .* mHat_k - alpha * weight_decay * w; % adamW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %w = w - learningRate*grad;

    %lossValue = loss(x1Label, x2Label, x1, x2);

    losses(i) = returnedLoss;

    if mod(i, 10^floor(log10(i))) == 0
        fprintf("step: %i ===== loss:  %10.10f \n",i, returnedLoss);
    end

    if returnedLoss < 10^-11
        break;
    end
end

figure;
plot(steps, losses)               
set(gca, 'YScale', 'log'); 
xlabel('Epoch')
ylabel('Loss value')
grid on;

%quick testc
q11 = 0.75;
q22 = -0.25;
%we expect x1 = q1 + q2 - 1 = -0.5 and x2 = q1 - q2 + 1 = 2

[x11, x22] = forward([q11, q22], w);
fprintf("Quick Verification: x1= %10.10f ===== x2=  %10.10f \n", x11, x22)

function [lossResult] = lossFromForwardVector(q, xLabels, w)
    [x1Vector, x2Vector] = forward(q, w);
    lossResult = loss(xLabels(:,1), xLabels(:,2), x1Vector, x2Vector);
end

function [x1, x2] = forward(q, w)
    global arch;
    num_layers = length(arch) - 1;
    x = q;
    idx = 1;

    for l = 1:num_layers
        n_in = arch(l);
        n_out = arch(l+1);

        W_size = n_in * n_out;
        W = reshape(w(idx : idx + W_size - 1), [n_in, n_out]);
        idx = idx + W_size;

        b = w(idx : idx + n_out - 1);
        idx = idx + n_out;
        
        if l < num_layers
            x = Activation.paraRelu(x * W + repmat(b ,length(q(:,1)),1));
        else 
            x = x * W + repmat(b ,length(q(:,1)),1);
        end
    end

    x1 = x(:,1);
    x2 = x(:,2);
end

function y = loss(x1Label, x2Label, x1, x2)
    y = mean((x1Label - x1).^2 + (x2Label - x2).^2);
end

function [gradient, loss] = calcGradientOfLossFunction(inputTrainingVector, outputLabelVector, w)

    espsilonValue = 1e-6;
    
    trainedWeights = w;

    gradient = zeros(size(trainedWeights)); 
    
    for i = 1:length(trainedWeights)
        % Perturb the current weight using the espsilonValue to the right
        trainedWeightsPerturbedRight = trainedWeights;
        trainedWeightsPerturbedRight(i) = trainedWeightsPerturbedRight(i) + espsilonValue;
        
        % Calculate loss values for wright-perturbed parameter
        [costFunctionPertubedRight]= lossFromForwardVector(inputTrainingVector,outputLabelVector, trainedWeightsPerturbedRight);
        
        % Perturb the parameter value by -espsilonValue (ie, to the left)
        trainedWeightsPerturbedLeft = trainedWeights;
        trainedWeightsPerturbedLeft(i) = trainedWeightsPerturbedLeft(i) - espsilonValue;
        
        % Calculate loss values for perturbed parameter
        [costFunctionPertubedLeft] = lossFromForwardVector(inputTrainingVector, outputLabelVector, trainedWeightsPerturbedLeft);
        
        % Approximate the gradient contribution 
        gradient(i) = (costFunctionPertubedRight - costFunctionPertubedLeft)... 
        / (2 * espsilonValue);
        %pick up the loss
        if(i == length(trainedWeights))
            loss = costFunctionPertubedLeft;
        end
    end
end
    