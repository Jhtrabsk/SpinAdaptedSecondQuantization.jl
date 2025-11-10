E_ai[:,:] = E_ai[:,:] .+  +0.25000000  * extract_mat(p3, "AI", o, v) * fixed_einsum("ia->ai", extract_mat(F, "ov", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -0.25000000  * fixed_einsum("Bia,B->ai", extract_mat(g_p, "VAov", o, v), extract_mat(p3, "VI", o, v), optimize="optimal");
