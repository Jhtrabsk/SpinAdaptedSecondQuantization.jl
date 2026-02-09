E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("Bai,Cai->CB", extract_mat(p, "AIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("Baibj,Caibj->CB", extract_mat(p2, "AIvovo", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
