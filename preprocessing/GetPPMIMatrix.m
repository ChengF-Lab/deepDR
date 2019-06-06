%% Calculate Positive Pointwise Mutual Information Matrix %%

function PPMI = GetPPMIMatrix(M)

M = ScaleSimMat(M);%Comment this step if you get every element in the matrix to be zero

[p, q] = size(M);
assert(p==q, 'M must be a square matrix!');

col = sum(M);
row = sum(M,2);

D = sum(col);
PPMI = log(D * M ./(row*col));
PPMI(PPMI<0)=0;
IdxNan = isnan(PPMI);
PPMI(IdxNan) = 0;

end