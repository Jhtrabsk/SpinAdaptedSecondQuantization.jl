E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("ia,bj->bjai", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("jb,ai->bjai", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("ai,jb->bjai", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IIov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("bj,ia->bjai", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IIov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("ci,cajb->bjai", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("cj,cbia->bjai", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("ak,ikjb->bjai", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("bk,iajk->bjai", extract_mat(c1, "vo", o, v), extract_mat(g, "ovoo", o, v), optimize="optimal");