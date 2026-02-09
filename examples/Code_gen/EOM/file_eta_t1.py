E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("ia->ai", extract_mat(F, "ov", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("ia->ai", extract_mat(g_p, "IIov", o, v), optimize="optimal");
