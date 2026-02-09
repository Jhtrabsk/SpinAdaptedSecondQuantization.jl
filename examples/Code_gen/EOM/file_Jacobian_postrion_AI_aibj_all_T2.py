E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +2.00000000  * extract_mat(h_p, "IA", o, v) * fixed_einsum("aibj->bjai", extract_mat(c2, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -2.00000000  * fixed_einsum("cibj,ca->bjai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAvv", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -2.00000000  * fixed_einsum("cjai,cb->bjai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAvv", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +2.00000000  * fixed_einsum("akbj,ik->bjai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAoo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +2.00000000  * fixed_einsum("aibk,jk->bjai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAoo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -4.00000000  * fixed_einsum("aibj,kk->bjai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAoo", o, v), optimize="optimal");
