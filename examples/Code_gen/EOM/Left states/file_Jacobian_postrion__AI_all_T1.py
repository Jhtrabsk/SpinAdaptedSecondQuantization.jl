E = E .+  -1.00000000  * fixed_einsum("ai,ai->", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IAvo", o, v), optimize="optimal");
E = E .+  -1.00000000  * fixed_einsum("ai,jb,aibj->", extract_mat(c1, "vo", o, v), extract_mat(g_p, "IAov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
