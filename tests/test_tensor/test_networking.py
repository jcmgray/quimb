import quimb.tensor as qtn


def test_istree():
    assert qtn.Tensor().as_network().istree()
    tn = qtn.rand_tensor([2] * 1, ["x"]).as_network()
    assert tn.istree()
    tn |= qtn.rand_tensor([2] * 3, ["x", "y", "z"])
    assert tn.istree()
    tn |= qtn.rand_tensor([2] * 2, ["y", "z"])
    assert tn.istree()
    tn |= qtn.rand_tensor([2] * 2, ["x", "z"])
    assert not tn.istree()


def test_isconnected():
    assert qtn.Tensor().as_network().isconnected()
    tn = qtn.rand_tensor([2] * 1, ["x"]).as_network()
    assert tn.isconnected()
    tn |= qtn.rand_tensor([2] * 3, ["x", "y", "z"])
    assert tn.isconnected()
    tn |= qtn.rand_tensor([2] * 2, ["w", "u"])
    assert not tn.isconnected()
    assert not (qtn.Tensor() | qtn.Tensor()).isconnected()


def test_get_path_between_tids():
    tn = qtn.MPS_rand_state(5, 3)
    path = tn.get_path_between_tids(0, 4)
    assert path.tids == (0, 1, 2, 3, 4)
    path = tn.get_path_between_tids(3, 0)
    assert path.tids == (3, 2, 1, 0)


def test_subgraphs():
    k1 = qtn.MPS_rand_state(6, 7, site_ind_id="a{}")
    k2 = qtn.MPS_rand_state(8, 7, site_ind_id="b{}")
    tn = k1 | k2
    s1, s2 = tn.subgraphs()
    assert {s1.num_tensors, s2.num_tensors} == {6, 8}


def test_gen_paths_loops():
    tn = qtn.TN2D_rand(3, 4, 2)
    loops = tuple(tn.gen_paths_loops())
    assert len(loops) == 6
    assert all(len(loop) == 4 for loop in loops)


def test_gen_paths_loops_intersect():
    tn = qtn.TN2D_empty(5, 4, 2)
    loops = tuple(tn.gen_paths_loops(8, False))
    na = len(loops)
    assert na == len(frozenset(loops))
    assert na == len(frozenset(map(frozenset, loops)))

    loops = tuple(tn.gen_paths_loops(8, True))
    nb = len(loops)
    assert nb == len(frozenset(loops))
    assert nb == len(frozenset(map(frozenset, loops)))
    assert nb > na


def test_gen_inds_connected():
    tn = qtn.TN2D_rand(3, 4, 2)
    patches = tuple(tn.gen_inds_connected(2))
    assert len(patches) == 34


def test_connected_bipartitions():
    tn = qtn.TN_rand_reg(6, 3, 2)
    for pa, pb in tn.connected_bipartitions():
        assert pa | pb == frozenset(tn.tensor_map)
        assert not (pa & pb)
        assert tn._select_tids(pa).isconnected()
        assert tn._select_tids(pb).isconnected()
