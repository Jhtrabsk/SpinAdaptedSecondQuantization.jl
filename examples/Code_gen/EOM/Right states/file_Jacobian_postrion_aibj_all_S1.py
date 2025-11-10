E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Aai,Abj->bjai", extract_mat(g_p, "IVvo", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Abj,Aai->bjai", extract_mat(g_p, "IVvo", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Aci,akbj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Acj,aibk->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Aak,cibj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Abk,cjai->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Akc,Aai,ckbj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Akc,Abj,ckai->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
