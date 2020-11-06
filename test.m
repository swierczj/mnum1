n = 10;
A = zeros(n);
A_2 = zeros(n);
b = zeros(n, 1);
b_2 = zeros(n, 1);
%x = zeros(n, 1);
% matrices init
for i = 1:n
    for j = 1:n
        if i == j
            A(i, j) = 6;
        elseif i == j-1 || i == j+1
            A(i, j) = 2;
        end
        A_2(i, j) = 4/(5*(i + j - 1));
    end
    
    b(i) = 9 + 0.5*i;
    if mod(i, 2)
        b_2(i) = 1/(2*i);
    else
        b_2(i) = 0;
    end    
%     x(i) = i;
end

result_1 = gaussian_elimination_complete_pivoting(A, b, n);
result_2 = gaussian_elimination_complete_pivoting(A_2, b_2, n);
% residuum
r_1 = A*result_1 - b;
r_2 = A_2*result_2 - b_2;

r_1 = abs(r_1);
r_2 = abs(r_2);

% plot TODO  
%plot(n, sum(r_2), 'o');

% [L_1, D_1, U_1] = decompose_matrix(A, n);
test_1 = gauss_seidel_method([16, 3; 7, -11], [11; 13], 2);
test_2 = gauss_seidel_method(A, b, n);


function solution = gaussian_elimination_complete_pivoting(A, b, n)
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

end

function gs_solution = gauss_seidel_method(A, b, n)
    max_iters = 10;
    x = zeros(n, 1);
    [L, D, U] = decompose_matrix(A, n);
    L_prim = D + L;
    L_prim = L_prim^(-1);
    for iter = 1:max_iters
        second = b - U*x;
        x = L_prim*second;
    end
    gs_solution = x;
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