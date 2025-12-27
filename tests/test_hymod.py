import torch
from project.hydro_selection.models.layers.jit_core import hymod_timestep_loop


def test_hymod_basic_shapes_and_nonneg():
    T = 10
    B = 2
    E = 3
    Nq = 2

    P = torch.ones((T, B, E), dtype=torch.float32)
    Tavg = torch.full((T, B, E), 5.0)
    PET = torch.full((T, B, E), 0.5)

    snow_store = torch.zeros((B, E))
    XHuz = torch.zeros((B, E))
    XCuz = torch.zeros((B, E))
    Xs = torch.zeros((B, E))
    Xq = torch.zeros((B, E, Nq))

    # parameters: use constants broadcastable to (B,E)
    Tth = torch.full((B, E), 0.0)
    Tb = torch.full((B, E), 0.0)
    DDF = torch.full((B, E), 0.2)
    Huz = torch.full((B, E), 10.0)
    Cpar = torch.full((B, E), 5.0)
    Bpar = torch.full((B, E), 0.5)
    Kv = torch.full((B, E), 0.5)
    alpha = torch.full((B, E), 0.6)
    Kq = torch.full((B, E), 0.3)
    Ks = torch.full((B, E), 2.0)

    outputs = hymod_timestep_loop(
        P, Tavg, PET, snow_store, XHuz, XCuz, Xs, Xq,
        Tth, Tb, DDF, Huz, Cpar, Bpar, Kv, alpha, Kq, Ks, 1e-6
    )

    # unpack some outputs
    snow_out, melt_out, effPrecip_out, PE_out, OV_out, AE_out, OV1_out, OV2_out, Qq_out, Qs_out, Q_out, *states = outputs

    assert snow_out.shape == (T, B, E)
    assert melt_out.shape == (T, B, E)
    assert effPrecip_out.shape == (T, B, E)
    assert Q_out.shape == (T, B, E)

    # non-negativity
    assert torch.all(Q_out >= -1e-6)
    assert torch.all(AE_out >= -1e-6)

    # states returned
    snow_store_f, XHuz_f, XCuz_f, Xs_f, Xq_f = states
    assert Xq_f.shape == (B, E, Nq)


if __name__ == "__main__":
    test_hymod_basic_shapes_and_nonneg()
    print("hymod basic test passed")
