function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
num_train_Y = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    thetaList = zeros(1, m_normal_train_X);
    for j = 1 : m_normal_train_X,
        temp = 0;
        for i = 1 : num_train_Y,
            temp1 = 0;
            for k = 1 : m_normal_train_X,
                temp1 += theta(k) * X(i, k);
            end;
            temp += (temp1 - y(i)) * (X(i, j));
        end;
        thetaList(j) = theta(j) - alpha / num_train_Y * temp;
    end;
    for (j = 1 : m_normal_train_X),
        theta(j) = thetaList(j);
    end;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
