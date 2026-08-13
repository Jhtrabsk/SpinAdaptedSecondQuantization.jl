E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("aibj,Caibj->CB", extract_mat(L3, "AIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal");
E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("aibj,Cbjai->CB", extract_mat(L3, "AIvovo", o, v), extract_mat(R3, "VIvovo", o, v), optimize="optimal");
