E_jai[:,:] = E_jai[:,:] .+  -0.50000000  * fixed_einsum("ji->jai", extract_mat(F, "oo", o, v), optimize="optimal");
E_jai[:,:] = E_jai[:,:] .+  +0.50000000  * fixed_einsum("aija->jai", extract_mat(L, "voov", o, v), optimize="optimal");
E_jai[:,:] = E_jai[:,:] .+  +0.50000000  * fixed_einsum("ji->jai", extract_mat(g_p, "IIoo", o, v), optimize="optimal");
E_jai[:,:] = E_jai[:,:] .+  +0.50000000  * fixed_einsum("kbja,aibk->jai", extract_mat(L, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_jai[:,:] = E_jai[:,:] .+  -0.50000000  * fixed_einsum("kbjc,bkci->jai", extract_mat(g, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
