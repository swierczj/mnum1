% Change parameters for different algorithms, datasets and max time for
% algorithm 
% true, true - Gaussian elimination with complete pivoting, 1st dataset 
% false, true - Gauss-Seidel mathod 1st dataset
% true, false - Gaussian elimination with complete pivoting, 2nd dataset
% false, false - Gauss-Seidel method 2nd dataset
max_time = 2 * 60; % time in seconds
data = perform_analysis(false, false, max_time);
figure
plot(data(:, 2), data(:, 1));
title('Zależność residuum od ilości równań dla A, G-S')
xlabel('Liczba równań')
ylabel('Norma residuum')

function residue_data = perform_analysis(is_gecp, is_first, max_time)
    n = 5;
    elapsed_time = 0;
    residue_data = [];
    while elapsed_time <= max_time
        A = zeros(n);
        b = zeros(n, 1);
        % matrices init
        for i = 1:n
            for j = 1:n
                if is_first
                    if i == j
                        A(i, j) = 6;
                    elseif i == j-1 || i == j+1
                        A(i, j) = 2;
                    end
                else
                    A(i, j) = 4/(5*(i + j - 1));    
                end
            end
            if is_first
                b(i) = 9 + 0.5*i;
            else    
                if mod(i, 2)
                    b(i) = 1/(2*i);
                else
                    b(i) = 0;
                end
            end
        end
        
        fprintf('Dataset first: %d, condition number: %d\n', is_first, get_condition_number(A));
        if is_gecp
            [result, elapsed_time] = gaussian_elimination_complete_pivoting(A, b, n);
        else
            fprintf('Diagonally dominant: %d\n', is_diagonally_dominant(A));
            [result, elapsed_time] = gauss_seidel_method(A, b, n);
        end    
        
        residue = A*result - b;
        residue_norm = norm(residue);
        residue_data = [residue_data; [residue_norm, n]];
        fprintf('Current n: %d, elapsed time: %d\n', n, elapsed_time);
        n = n*2;
    end
end

function [solution, elapsed_time] = gaussian_elimination_complete_pivoting(A, b, n)
    tic
    x = 1:n;
    x = x(:);
    for k = 1:n
        tmp = A(k:n, k:n);
        [biggest_vals, row_indices] = max(abs(tmp));
        [~, val_col] = max(biggest_vals);
        val_row = row_indices(val_col);
        % columns switch
        P_cols = get_permutation_matrix(n, get_permutation_indices(n, [k, val_col + k-1]));
        A = A*P_cols;
        x = P_cols*x;
        % rows switch
        P_rows = get_permutation_matrix(n, get_permutation_indices(n, [k, val_row + k-1]));
        A = P_rows*A;
        b = P_rows*b;
        L = get_lower_matrix(n, k, A(k:end, k));
        A = L*A;
        b = L*b;
    end

    result = (A^-1)*b;
    result_permutation = get_permutation_matrix(n, x);
    solution = result_permutation*result;
    elapsed_time = toc;
end

function [gs_solution, elapsed_time] = gauss_seidel_method(A, b, n)
    tic
    eps = 10^(-8);
    max_iters = 1000;
    current_iter = 0;
    x = rand(n, 1);
    [L, D, U] = decompose_matrix(A, n);
    L_prime = D + L;
    residue = A*x - b;
    while norm(residue) > eps && current_iter < max_iters % check convergence
        x = L_prime^(-1)*(b - U*x);
        residue = A*x - b;
        current_iter = current_iter + 1;
    end
    fprintf('Iterations: %d, residue norm: %d\n', current_iter, norm(residue));
    gs_solution = x;
    elapsed_time = toc;
end

function diagonally_dominant = is_diagonally_dominant(A)
    diagonally_dominant = true;
    i = 1;
    while i <= size(A, 1) && diagonally_dominant
        diagonally_dominant = abs(A(i, i)) >= sum(abs(A(i, :))) - abs(A(i, i));
        i = i+1;
    end
end

function condition_number = get_condition_number(A)
    max_norm = 0;
    max_inv_norm = 0;
    inv_matrix = A^(-1);
    rows_number = size(A, 1);
    for i = 1:rows_number
        current_norm = sum(abs(A(i, :)));
        if current_norm > max_norm
            max_norm = current_norm;
        end
        current_inv_norm = sum(abs(inv_matrix(i, :)));
        if current_inv_norm > max_inv_norm
            max_inv_norm = current_inv_norm;
        end
    end
    condition_number = max_norm * max_inv_norm;
end

function [L, D, U] = decompose_matrix(M, size)
    L = zeros(size);
    U = zeros(size);
    D = zeros(size);
    for i = 1:size
        for j = 1:size
            if j == i
                D(i, j) = M(i, j);
            elseif j > i
                U(i, j) = M(i, j);
            else
                L(i, j) = M(i, j);
            end    
        end    
    end
end

function P = get_permutation_matrix(size, to_swap_indices)
    P = eye(size);
    for i = 1:numel(to_swap_indices)
        new_position = to_swap_indices(i);
        if i ~= new_position
            P(new_position, i) = 1;
            P(i, i) = 0;
        end    
    end
end

function L = get_lower_matrix(size, column_idx, column_values)
    L = eye(size);
    step_number = column_idx;
    l = -column_values./column_values(1);
    L(step_number+1:end, column_idx) = l(2:end);
end

function indices = get_permutation_indices(size, indices_to_swap)
    indices = 1:size;
    if indices_to_swap(1) ~= indices_to_swap(2)
        indices(indices_to_swap(1)) = indices_to_swap(2);
        indices(indices_to_swap(2)) = indices_to_swap(1);
    end
end