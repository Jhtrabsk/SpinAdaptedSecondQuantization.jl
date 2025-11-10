E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("aijb->bjai", extract_mat(L, "voov", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("kcjb,aick->bjai", extract_mat(L, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
