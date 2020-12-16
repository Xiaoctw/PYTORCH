#
# 计算PPMI矩阵的MATLAB方法，列向量乘行向量会得到一个矩阵
#
# function PPMI = GetPPMIMatrix(M)
#
# M = ScaleSimMat(M);
#
# [p, q] = size(M);
# assert(p==q, 'M must be a square matrix!');
#
# col = sum(M);
# row = sum(M,2);
#
# D = sum(col);
# PPMI = log(D * M ./(row*col));
# PPMI(PPMI<0)=0;
# IdxNan = isnan(PPMI);
# PPMI(IdxNan) = 0;
#
# end