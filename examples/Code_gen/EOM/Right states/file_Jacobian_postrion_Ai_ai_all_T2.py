E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("bjai,jb->ai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "AIov", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +1.00000000  * fixed_einsum("biaj,jb->ai", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "AIov", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("jbkc,bjci,ak->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("jbkc,bjak,ci->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +2.00000000  * fixed_einsum("jbkc,bjai,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -1.00000000  * fixed_einsum("jbkc,biaj,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
