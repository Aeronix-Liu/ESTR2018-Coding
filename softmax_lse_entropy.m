function softmax_lse_entropy()
% SOFTMAX_LSE_ENTROPY
%   Visualize, for 2-dimensional inputs x = [x1, x2],
%   - the log-sum-exp value:      LSE(x) = log(exp(x1) + exp(x2))
%   - the negative entropy of the corresponding softmax distribution:
%         p = softmax(x),  H(p) = -sum_i p_i * log(p_i),
%         we plot  -H(p)  (larger -> more certain distribution).
%
%   This helps illustrate how the raw 2D input relates to:
%   - a smooth "log-partition" (log-sum-exp),
%   - and the certainty / peakedness of the softmax (via -entropy).
%
%   Run in MATLAB with:
%       >> softmax_lse_entropy

    % Define a 2D grid of inputs
    x_min = -4;
    x_max =  4;
    n_points = 201;

    x1_vals = linspace(x_min, x_max, n_points);
    x2_vals = linspace(x_min, x_max, n_points);
    [X1, X2] = meshgrid(x1_vals, x2_vals);

    % Compute log-sum-exp and negative entropy over the grid
    LSE = zeros(size(X1));
    negH = zeros(size(X1));

    for i = 1:n_points
        for j = 1:n_points
            x1 = X1(i, j);
            x2 = X2(i, j);

            % Log-sum-exp: log(exp(x1) + exp(x2))
            % Use a numerically stable version:
            m = max([x1, x2]);
            LSE(i, j) = m + log(exp(x1 - m) + exp(x2 - m));

            % Softmax probabilities
            ex1 = exp(x1 - m);
            ex2 = exp(x2 - m);
            Z   = ex1 + ex2;
            p1  = ex1 / Z;
            p2  = ex2 / Z;

            % Entropy H(p) = -sum p_i log p_i
            % Handle p*log(p) = 0 when p = 0 using safe helper
            H = - (safe_p_log_p(p1) + safe_p_log_p(p2));

            % Negative entropy: -H
            negH(i, j) = -H;
        end
    end

    % Plot: left = LSE, right = negative entropy
    figure('Name', '2D input vs. log-sum-exp and negative entropy', ...
           'Color', 'w');

    % --- Plot log-sum-exp ---
    subplot(1, 2, 1);
    surf(X1, X2, LSE, 'EdgeColor', 'none');
    colormap(parula);
    colorbar;
    % Make axes and labels bold and readable (without darkening the cube)
    set(gca, 'LineWidth', 1.2, 'FontWeight', 'bold', 'FontSize', 12);
    xlabel('x_1', 'FontWeight', 'bold', 'FontSize', 13);
    ylabel('x_2', 'FontWeight', 'bold', 'FontSize', 13);
    zlabel('LSE(x) = log(exp(x_1) + exp(x_2))', ...
           'FontWeight', 'bold', 'FontSize', 13);
    title('Log-sum-exp over 2D input', 'FontWeight', 'bold', 'FontSize', 14);
    view(45, 30);

    % --- Plot negative entropy of softmax ---
    subplot(1, 2, 2);
    surf(X1, X2, negH, 'EdgeColor', 'none');
    colormap(parula);
    colorbar;
    % Same styling for the second subplot
    set(gca, 'LineWidth', 1.2, 'FontWeight', 'bold', 'FontSize', 12);
    xlabel('x_1', 'FontWeight', 'bold', 'FontSize', 13);
    ylabel('x_2', 'FontWeight', 'bold', 'FontSize', 13);
    zlabel('-H(softmax(x))', 'FontWeight', 'bold', 'FontSize', 13);
    title('Negative entropy of softmax over 2D input', ...
          'FontWeight', 'bold', 'FontSize', 14);
    view(45, 30);

end


function val = safe_p_log_p(p)
% SAFE_P_LOG_P   Compute p*log(p) with the convention 0*log(0) = 0.
    if p <= 0
        val = 0;
    else
        val = p * log(p);
    end
end


