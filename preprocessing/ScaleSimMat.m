%% Scale Similar Matrix by Row %%

function W = ScaleSimMat(W)

%scale 
W = W - diag(diag(W));  %diagonal elements must be 0
D = diag(sum(W), 0);    %degree matrix

W = D^(-1)*W;