E = E .+  +1.00000000  * fixed_einsum("Aai,ai->A", extract_mat(c1, "vo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E = E .+  +1.00000000  * fixed_einsum("aibj,Aaibj->A", extract_mat(c2, "vovo", o, v), extract_mat(s2, "AIvovo", o, v), optimize="optimal");
E = E .+  -1.00000000  * fixed_einsum("Baibj,Aai,Bbj->A", extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E = E .+  -1.00000000  * fixed_einsum("Baibj,Abj,Bai->A", extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
