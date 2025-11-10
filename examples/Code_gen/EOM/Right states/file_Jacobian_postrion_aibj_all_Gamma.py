E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Aki,A,akbj->bjai", extract_mat(g_p, "IVoo", o, v), extract_mat(p3, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akj,A,aibk->bjai", extract_mat(g_p, "IVoo", o, v), extract_mat(p3, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Aac,A,cibj->bjai", extract_mat(g_p, "IVvv", o, v), extract_mat(p3, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Abc,A,cjai->bjai", extract_mat(g_p, "IVvv", o, v), extract_mat(p3, "VI", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
