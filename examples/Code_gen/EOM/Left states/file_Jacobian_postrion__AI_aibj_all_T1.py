E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("ai,jb->bjai", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IAov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("bj,ia->bjai", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IAov", o, v), optimize="optimal");
