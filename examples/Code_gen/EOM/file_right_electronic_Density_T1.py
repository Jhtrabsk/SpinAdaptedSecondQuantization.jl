E_ai[:,:] +=  +2.00000000  * extract_mat(r1, "vo", o, v);
E_ji[:,:] = E_ji[:,:] .+  -1.00000000  * fixed_einsum("aj,ai->ji", extract_mat(l1, "vo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
E_ba[:,:] = E_ba[:,:] .+  +1.00000000  * fixed_einsum("ai,bi->ba", extract_mat(l1, "vo", o, v), extract_mat(r1, "vo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  +4.00000000  * fixed_einsum("bj,bjai->ai", extract_mat(l1, "vo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
E_ai[:,:] = E_ai[:,:] .+  -2.00000000  * fixed_einsum("bj,biaj->ai", extract_mat(l1, "vo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
