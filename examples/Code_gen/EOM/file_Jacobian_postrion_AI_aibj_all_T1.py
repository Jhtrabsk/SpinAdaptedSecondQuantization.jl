E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -2.00000000  * fixed_einsum("ia,bj->bjai", extract_mat(g_p, "IAov", o, v), extract_mat(l1, "vo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("ib,aj->bjai", extract_mat(g_p, "IAov", o, v), extract_mat(l1, "vo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("ja,bi->bjai", extract_mat(g_p, "IAov", o, v), extract_mat(l1, "vo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -2.00000000  * fixed_einsum("jb,ai->bjai", extract_mat(g_p, "IAov", o, v), extract_mat(l1, "vo", o, v), optimize="optimal");
