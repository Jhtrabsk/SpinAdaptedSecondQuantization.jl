E_bai[:,:] = E_bai[:,:] .+  +0.50000000  * fixed_einsum("ab->bai", extract_mat(F, "vv", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  +0.50000000  * fixed_einsum("ibai->bai", extract_mat(L, "ovvo", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  -0.50000000  * fixed_einsum("ab->bai", extract_mat(g_p, "IIvv", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  +0.50000000  * fixed_einsum("jcib,aicj->bai", extract_mat(L, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  -0.50000000  * fixed_einsum("jckb,akcj->bai", extract_mat(g, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
