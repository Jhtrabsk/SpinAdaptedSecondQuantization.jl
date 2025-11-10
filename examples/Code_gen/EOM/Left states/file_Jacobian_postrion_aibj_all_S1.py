E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Aia,Abj->bjai", extract_mat(g_p, "VIov", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Ajb,Aai->bjai", extract_mat(g_p, "VIov", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcia,Abj,Ack->bjai", extract_mat(L, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcjb,Aai,Ack->bjai", extract_mat(L, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kajb,Aci,Ack->bjai", extract_mat(g, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kbia,Acj,Ack->bjai", extract_mat(g, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("icjb,Aak,Ack->bjai", extract_mat(g, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("iajc,Abk,Ack->bjai", extract_mat(g, "ovov", o, v), extract_mat(p, "VIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
