E_CB[:,:] = E_CB[:,:] .+  +1.00000000  * fixed_einsum("ai,ai->CB", extract_mat(R2, "AIvo", o, v), extract_mat(l1, "vo", o, v), optimize="optimal");
E_CB[:,:] = E_CB[:,:] .+  -1.00000000  * fixed_einsum("Daibj,ai,Dbj->CB", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "AIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_CB[:,:] = E_CB[:,:] .+  -1.00000000  * fixed_einsum("Daibj,bj,Dai->CB", extract_mat(L3, "VIvovo", o, v), extract_mat(R2, "AIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
