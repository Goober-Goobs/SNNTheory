import torch
from . import snn

def smoke_test(verbose=True):
    """
    Full framework smoke test for customSNN.

    Returns:
        bool
    """

    try:

        # =====================================================
        # Dummy classes
        # =====================================================

        class Child(snn.Module):
            def __init__(self):
                super().__init__()
                self.w = snn.Parameter.ones(2)

            def forward(self, x):
                return x


        class ResetNeuron(snn.Module):
            def __init__(self):
                super().__init__()
                self.v = torch.ones(3)

            def reset_state(self):
                self.v.zero_()
                super().reset_state()

            def forward(self, x):
                return x


        class Dummy(snn.Module):
            def __init__(self):
                super().__init__()

                self.weight = snn.Parameter.randn(2, 3)
                self.state = torch.zeros(4)
                self.child = Child()
                self.neuron = ResetNeuron()

            def forward(self, x):
                return x


        # =====================================================
        # Instantiate
        # =====================================================

        m = Dummy()

        if verbose:
            print("\n[Smoke Test] Module created")
            print(m)


        # =====================================================
        # 1. Registration
        # =====================================================

        assert "weight" in m._parameters
        assert "state" in m._buffers
        assert "child" in m._modules
        assert "neuron" in m._modules


        # =====================================================
        # 2. Named parameters
        # =====================================================

        named_params = dict(m.named_parameters())
        assert "weight" in named_params
        assert "child.w" in named_params


        # =====================================================
        # 3. Named modules
        # =====================================================

        named_mods = dict(m.named_modules())
        assert "child" in named_mods
        assert "neuron" in named_mods


        # =====================================================
        # 4. State dict
        # =====================================================

        state = m.state_dict()
        assert "weight" in state
        assert "child.w" in state


        # =====================================================
        # 5. Load state dict
        # =====================================================

        backup = m.state_dict()
        m.weight.data += 10
        m.load_state_dict(backup)

        assert torch.allclose(
            m.state_dict()["weight"],
            backup["weight"]
        )


        # =====================================================
        # 6. Device movement
        # =====================================================

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m.to(device)

        for _, p in m.named_parameters():
            assert p.data.device == device

        for _, b in m.named_buffers():
            assert b.device == device


        # =====================================================
        # 7. Reset state
        # =====================================================

        m.neuron.v += 5
        m.reset_state()
        assert torch.all(m.neuron.v == 0)


        # =====================================================
        # 8. Forward call
        # =====================================================

        assert m(123) == 123


        # =====================================================
        # 9. Traversal
        # =====================================================

        mods = list(m.modules())
        assert m in mods
        assert m.child in mods

        params = list(m.parameters())
        assert any(p is m.weight for p in params)

        buffers = list(m.buffers())
        assert any(torch.equal(b, m.state) for b in buffers)


        # =====================================================
        # SUCCESS
        # =====================================================

        if verbose:
            print("\n✅ Smoke test PASSED")

        return True


    except Exception as e:

        print("\n❌ Smoke test FAILED")
        print("Error:", repr(e))

        return False


# =========================================================
# Allow running directly
# =========================================================

if __name__ == "__main__":
    ok = smoke_test(verbose=True)

    raise SystemExit(0 if ok else 1)
