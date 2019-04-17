function [C] = strassen(A,B,bc)
% Strassen's matrix multiplication algorithm
% bc is the base case dimension
% returns C=A*B for any A,B with compatible dimensions
% returns cnt which is the operation count

    % check dimensions
    [m,n] = size(A);
    [nn,p] = size(B);
    if n ~= nn,
        error('incompatible dimensions');
    end
    C = zeros(m,p);
    
    % base case
    if m <= bc || n <= bc || p <= bc,
        C = A*B;
        return;
    end
    
    % dynamic peeling (handles case of odd dimensions) 
    if mod(m,2) == 1,
        [C(1:m-1,:)] = strassen(A(1:m-1,:),B,bc);
        C(m,:) = A(m,:)*B;
        return;
    end
    if mod(n,2) == 1,
        C = strassen(A(:,1:n-1),B(1:n-1,:),bc);
        C = C + A(:,n)*B(n,:);
        return;
    end
    if mod(p,2) == 1,
        C(:,1:p-1) = strassen(A,B(:,1:p-1),bc);
        C(:,p) = A*B(:,p);
        return;
    end 
    
    % divide (m,n,p all even)
    m2 = m/2; n2 = n/2; p2 = p/2;
    A11 = A(1:m2,1:n2);   A12 = A(1:m2,n2+1:n);
    A21 = A(m2+1:m,1:n2); A22 = A(m2+1:m,n2+1:n);
    B11 = B(1:n2,1:p2);   B12 = B(1:n2,p2+1:p);
    B21 = B(n2+1:n,1:p2); B22 = B(n2+1:n,p2+1:p);
    
    % conquer
    M1 = strassen(A11    ,B12-B22,bc);
    M2 = strassen(A11+A12,B22    ,bc);
    M3 = strassen(A21+A22,B11    ,bc);
    M4 = strassen(A22    ,B21-B11,bc);
    M5 = strassen(A11+A22,B11+B22,bc);
    M6 = strassen(A12-A22,B21+B22,bc);
    M7 = strassen(A11-A21,B11+B12,bc);
    
    % combine
    C(1:m2,1:p2)     = M5+M4-M2+M6;
    C(1:m2,p2+1:p)   = M1+M2;
    C(m2+1:m,1:p2)   = M3+M4;
    C(m2+1:m,p2+1:p) = M1+M5-M3-M7;
    
end