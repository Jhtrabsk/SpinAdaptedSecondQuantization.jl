E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("ai,Cai->CB", extract_mat(L2, "AIvo", o, v), extract_mat(R2, "VIvo", o, v), optimize="optimal");
