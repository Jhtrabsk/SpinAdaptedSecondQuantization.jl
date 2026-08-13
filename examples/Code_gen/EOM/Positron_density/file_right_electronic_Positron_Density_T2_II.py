E_CB[:,:] = E_CB[:,:] .+  +2.00000000  * fixed_einsum("aibj,aibj->", extract_mat(l2, "vovo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
