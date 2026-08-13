E_AI= E_AI .+  +1.00000000  * fixed_einsum("Aai,ai->A", extract_mat(L2, "AIvo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
