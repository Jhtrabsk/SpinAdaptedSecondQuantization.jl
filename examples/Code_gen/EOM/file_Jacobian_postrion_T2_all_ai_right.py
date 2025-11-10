E_ai[:,:] = E_ai[:,:] .+  +4.00000000  * fixed_einsum("jb,bjai->ai", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("jb,biaj->ai", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -4.00000000  * fixed_einsum("bjai,jb->ai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IIov", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("biaj,jb->ai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IIov", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("jbki,bjak->ai", extract_mat(L, "ovoo", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("jbac,bjci->ai", extract_mat(L, "ovvv", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
