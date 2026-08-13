E_II = E_II .+  +1.00000000  * fixed_einsum("ai,ai->", extract_mat(l1, "vo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
E_II = E_II .+  -1.00000000  * fixed_einsum("Daibj,ai,Dbj->", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_II = E_II .+  -1.00000000  * fixed_einsum("Daibj,bj,Dai->", extract_mat(L3, "VIvovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
