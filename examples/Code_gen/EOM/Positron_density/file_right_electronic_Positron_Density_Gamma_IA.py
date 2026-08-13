E_CB[:,:] +=  +1.00000000  * extract_mat(R1, "AI", o, v);
E_CB[:,:] = E_CB[:,:] .+  -1.00000000  * extract_mat(R1, "AI", o, v) * fixed_einsum("Dai,Dai->", extract_mat(L2, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_CB[:,:] = E_CB[:,:] .+  -1.00000000  * extract_mat(R1, "AI", o, v) * fixed_einsum("Daibj,Daibj->", extract_mat(L3, "VIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
