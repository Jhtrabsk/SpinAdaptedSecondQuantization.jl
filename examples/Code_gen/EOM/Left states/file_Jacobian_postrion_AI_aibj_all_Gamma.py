E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * extract_mat(p3, "AI", o, v) * fixed_einsum("iajb->bjai", extract_mat(g, "ovov", o, v), optimize="optimal");
