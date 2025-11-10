E_bai[:,:] = E_bai[:,:] .+  +0.50000000  * fixed_einsum("ib->bai", extract_mat(F, "ov", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  -0.50000000  * fixed_einsum("iiib->bai", extract_mat(g, "ooov", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  +0.50000000  * fixed_einsum("iaab->bai", extract_mat(g, "ovvv", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  +0.50000000  * fixed_einsum("ibaa->bai", extract_mat(g, "ovvv", o, v), optimize="optimal");
E_bai[:,:] = E_bai[:,:] .+  -0.50000000  * fixed_einsum("ib->bai", extract_mat(g_p, "IIov", o, v), optimize="optimal");
