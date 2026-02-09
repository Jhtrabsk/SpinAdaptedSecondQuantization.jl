E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +2.00000000  * fixed_einsum("iajb->bjai", extract_mat(L, "ovov", o, v), optimize="optimal");
