E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Aia,A->ai", extract_mat(g_p, "VIov", o, v), extract_mat(p3, "VI", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("jbia,A,Abj->ai", extract_mat(L, "ovov", o, v), extract_mat(p3, "VI", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
