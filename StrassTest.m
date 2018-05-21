

A = rand(1024,1024);
B = rand(1024,1024);

tic
strassen(A,B, 900);
toc

tic
C = A*B;
toc