E_ia[:,:] = E_ia[:,:] .+  +1.00000000  * fixed_einsum("ai->ia", extract_mat(c1, "vo", o, v), optimize="optimal");
