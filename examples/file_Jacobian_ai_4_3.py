E_ai[:,:] = E_ai[:,:] .+  -0.50000000  * fixed_einsum("ii->ai", extract_mat(F, "oo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +0.50000000  * fixed_einsum("aa->ai", extract_mat(F, "vv", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +0.50000000  * fixed_einsum("iaai->ai", extract_mat(L, "ovvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +0.50000000  * fixed_einsum("ii->ai", extract_mat(g_p, "IIoo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -0.50000000  * fixed_einsum("aa->ai", extract_mat(g_p, "IIvv", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +0.50000000  * fixed_einsum("jbia,aibj->ai", extract_mat(L, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -0.50000000  * fixed_einsum("jakb,ajbk->ai", extract_mat(g, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -0.50000000  * fixed_einsum("jbic,bjci->ai", extract_mat(g, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
