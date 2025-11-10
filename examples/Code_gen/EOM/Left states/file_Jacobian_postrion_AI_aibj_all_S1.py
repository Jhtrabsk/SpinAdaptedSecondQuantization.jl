E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("ia,bj->bjai", extract_mat(F, "ov", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("jb,ai->bjai", extract_mat(F, "ov", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("cajb,ci->bjai", extract_mat(g, "vvov", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("cbia,cj->bjai", extract_mat(g, "vvov", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("ikjb,ak->bjai", extract_mat(g, "ooov", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("iajk,bk->bjai", extract_mat(g, "ovoo", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("Bia,Bbj->bjai", extract_mat(g_p, "VAov", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("Bjb,Bai->bjai", extract_mat(g_p, "VAov", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("iajb,Bck,Bck->bjai", extract_mat(g, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VAvo", o, v), optimize="optimal");
