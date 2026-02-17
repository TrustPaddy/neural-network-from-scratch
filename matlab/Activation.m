classdef Activation
    methods (Static)
        % ReLU
        function y = relu(x)
            y = zeroes(size(x));
            y = max(0, x);
        end

        % Parametric ReLu
        function y = paraRelu(x)
            y = zeros(size(x));         % gleiche Größe wie x
            y(x > 0) = 1 * x(x > 0);  % wende Faktor 0.7 auf positive Elemente an
            y(x <= 0) = 0.1 * x(x <= 0);% wende Faktor 0.3 auf nicht-positive Elemente an
            % if(x > 0)
            %     y = 0.7 * x;
            % else
            %     y = 0.3 * x; 
            % end
        end

        % Leaky ReLU
        function y = leakyRelu(x, alpha)
            if nargin < 2
                alpha = 0.01;
            end
            y = max(alpha * x, x);
        end

        % Sigmoid
        function y = sigmoid(x)
            y = 1 ./ (1 + exp(-x));
        end

        % Tanh
        function y = tanh(x)
            %y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x)); 
            y = tanh(x);
        end

        % ELU
        function y = elu(x, alpha)
            if nargin < 2
                alpha = 1.0;
            end
            y = x;
            y(x < 0) = alpha * (exp(x(x < 0)) - 1);
        end

        % Softmax 
        function y = softmax(x)
            x = x - max(x, [], 2);  
            ex = exp(x);
            y = ex ./ sum(ex, 2);
        end
    end
end