E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Aai,A->ai", extract_mat(g_p, "IVvo", o, v), extract_mat(p3, "VI", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Ajb,A,bjai->ai", extract_mat(g_p, "IVov", o, v), extract_mat(p3, "VI", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
