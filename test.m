n = 100;
A = zeros(n);
b = zeros(n, 1);
x = zeros(n, 1);
% matrices init
for i = 1:n
    for j = 1:n
        if i == j
            A(i, j) = 6;
        elseif i == j-1 || i == j+1
            A(i, j) = 2;
        end
    end
    b(i) = 9 + 0.5*i;
    x(i) = i;
end
for k = 1:n
    tmp = A(k:n, k:n);
    [biggest_vals, row_indices] = max(abs(tmp));
    [biggest_val, val_col] = max(biggest_vals);
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
result = result_permutation*result;

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