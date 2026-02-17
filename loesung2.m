clc; clear; close all;

number_of_epochs = 10^5;

q1 = myRandVector(70, -1, 1);
q2 = myRandVector(70, -1, 1);
q = [q1, q2];

x1Label = (q1 + q2.*q2 - 1);
x2Label = (q1 - q2 + 1);
xLabel = [x1Label, x2Label];

arch = [2, 2, 4, 2];  

w_size = sum(arch(1:end-1).*arch(2:end))+sum(arch)-arch(1);

w = myRandVector(w_size, -1, 1)';

%learningRate = 0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AdamW %%%%%%%%%%%%%%%%%%%%
alpha = 0.001; % = r
mBeta_1 = 0.9;    % decay rate w.r.t first moment
mBeta_2 = 0.999;  % decay rate w.r.t second moment 
epsilon = 1e-8; % against zero division
weight_decay = 0.0; % <<< ADAMW-spefific
mk = zeros(size(w)); % first moment estimate
v_k = zeros(size(w)); % second moment estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

steps = 1:number_of_epochs;
losses = zeros(1, number_of_epochs);

tic;

for i = 1:number_of_epochs

    [x1, x2] = forward(q, w, arch);

    [grad, returnedLoss] = calcGradientOfLossFunction(q, xLabel, w, arch);

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
    %w = w - rk.* mHat_k; %instead of wk = wk - r*grad; adam 
    w = w - rk .* mHat_k - alpha * weight_decay * w; % adamW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %w = w - learningRate*grad;

    %lossValue = loss(x1Label, x2Label, x1, x2);

    losses(i) = returnedLoss;

    if mod(i, 10^floor(log10(i))) == 0
        fprintf("step: %i ===== loss:  %10.10f \n", i, returnedLoss);
    end

    if returnedLoss < 10^-6
        break;
    end
end

timerval = toc;
minutes = floor(timerval / 60);
seconds = floor(mod(timerval, 60));
milliseconds = round((timerval - floor(timerval)) * 1000);

figure;
plot(steps, losses)               
set(gca, 'YScale', 'log'); 
xlabel('Epoch')
ylabel('Loss value')
grid on;

%quick test
q11 = 0.75;
q22 = -0.25;
%we expect x1 = q1 + q2.*q2 - 1 = -0.1875 and x2 = q1 - q2 + 1= 2

[x11, x22] = forward([q11, q22], w, arch);
fprintf("Quick Verification: x1= %10.10f ===== x2= %10.10f \n", x11, x22)
fprintf("Elapsed Time %d min, %d s, %d ms", minutes, seconds, milliseconds)

function [lossResult] = lossFromForwardVector(q, xLabels, w, arch)
    [x1Vector, x2Vector] = forward(q, w, arch);
    lossResult = loss(xLabels(:,1), xLabels(:,2), x1Vector, x2Vector);
end

function [x1, x2] = forward(q, w, architecture)
    num_layers = length(architecture) - 1;
    x = q;
    idx = 1;

    for l = 1:num_layers
        n_in = architecture(l);
        n_out = architecture(l+1);

        W_size = n_in * n_out;
        W = reshape(w(idx : idx + W_size - 1), [n_in, n_out]);
        idx = idx + W_size;

        b = w(idx : idx + n_out - 1);
        idx = idx + n_out;

        
        % if l == 1 && l < 4
        %     x = Activation.sigmoid(x * W + repmat(b ,length(q(:,1)),1));
        if l == 1
            x = Activation.paraRelu(x * W + repmat(b ,length(q(:,1)),1));
        elseif l == num_layers
            x = x * W + repmat(b ,length(q(:,1)),1);
        else 
            x = Activation.tanh(x * W + repmat(b ,length(q(:,1)),1));
        end
    end

    x1 = x(:,1);
    x2 = x(:,2);
end

% MSE
function y = loss(x1Label, x2Label, x1, x2)
    y = mean((x1Label - x1).^2 + (x2Label - x2).^2);
end

function [gradient, loss] = calcGradientOfLossFunction(inputTrainingVector, outputLabelVector, w, arch)

    espsilonValue = 1e-6;
    
    trainedWeights = w;

    gradient = zeros(size(trainedWeights)); 
    
    for i = 1:length(trainedWeights)
        % Perturb the current weight using the espsilonValue to the right
        trainedWeightsPerturbedRight = trainedWeights;
        trainedWeightsPerturbedRight(i) = trainedWeightsPerturbedRight(i) + espsilonValue;
        
        % Calculate loss values for wright-perturbed parameter
        [costFunctionPertubedRight]= lossFromForwardVector(inputTrainingVector,outputLabelVector, trainedWeightsPerturbedRight, arch);
        
        % Perturb the parameter value by -espsilonValue (ie, to the left)
        trainedWeightsPerturbedLeft = trainedWeights;
        trainedWeightsPerturbedLeft(i) = trainedWeightsPerturbedLeft(i) - espsilonValue;
        
        % Calculate loss values for perturbed parameter
        [costFunctionPertubedLeft] = lossFromForwardVector(inputTrainingVector, outputLabelVector, trainedWeightsPerturbedLeft, arch);
        
        % Approximate the gradient contribution 
        gradient(i) = (costFunctionPertubedRight - costFunctionPertubedLeft)... 
        / (2 * espsilonValue);
        %pick up the loss
        if(i == length(trainedWeights))
            loss = costFunctionPertubedLeft;
        end
    end
end
