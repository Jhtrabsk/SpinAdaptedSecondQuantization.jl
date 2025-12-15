E_ai[:,:] +=  +1.00000000  * extract_mat(c1, "vo", o, v) * extract_mat(h_p, "IA", o, v);
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("bi,ba->ai", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IAvv", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("aj,ij->ai", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IAoo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * extract_mat(c1, "vo", o, v) * fixed_einsum("jj->", extract_mat(g_p, "IAoo", o, v), optimize="optimal");
