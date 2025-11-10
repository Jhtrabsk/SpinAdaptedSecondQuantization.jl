E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Ajb,Abjai->ai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +0.50000000  * fixed_einsum("Ajb,Abiaj->ai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +0.50000000  * fixed_einsum("Ajb,Aajbi->ai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Ajb,Aaibj->ai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
