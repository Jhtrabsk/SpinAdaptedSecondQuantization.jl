E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("ia->ai", extract_mat(g_p, "IAov", o, v), optimize="optimal");
