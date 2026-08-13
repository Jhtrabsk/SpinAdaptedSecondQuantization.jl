E_CB[:,:] = E_CB[:,:] .+  +2.00000000  * fixed_einsum("aibj,aibj->CB", extract_mat(R3, "AIvovo", o, v), extract_mat(l2, "vovo", o, v), optimize="optimal");
