E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("A,Aai->ai", extract_mat(h_p, "IV", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("Ajj,Aai->ai", extract_mat(g_p, "IVoo", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("Aji,Aaj->ai", extract_mat(g_p, "IVoo", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("Aab,Abi->ai", extract_mat(g_p, "IVvv", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
