import pytest

from quimb.tensor.environments import (
    EnvironmentPlan,
    all_blocks,
    find_1d_block,
)


def execute_symbolically(plan, blocks):
    cache = {}
    outputs = {}
    peak_cache = 0

    for move in plan.get_moves(blocks):
        if move.kind == "init":
            cache[move.output_env] = move.input_sites
        elif move.kind == "contract":
            represented = tuple(
                site for env_key in move.input_envs for site in cache[env_key]
            )
            for site in move.input_sites:
                assert site not in represented
                represented += (site,)
            cache[move.output_env] = represented
        elif move.kind == "output":
            outputs[move.output_block] = tuple(
                site for env_key in move.input_envs for site in cache[env_key]
            )
        else:
            del cache[move.input_envs[0]]
        peak_cache = max(peak_cache, len(cache))

    return outputs, peak_cache


class TestEnvironmentPlan:
    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    @pytest.mark.parametrize("cyclic", [False, True])
    def test_all_blocks(self, cyclic, schedule):
        for L in range(1, 12):
            for block_size in range(1, L + 1):
                blocks = all_blocks(L, block_size, cyclic)
                plan = EnvironmentPlan(L, cyclic=cyclic, schedule=schedule)
                outputs, _ = execute_symbolically(plan, blocks)
                created = [
                    move.output_env
                    for move in plan.get_moves(blocks)
                    if move.output_env is not None
                ]

                # a single block size is output in sorted order
                assert tuple(outputs) == blocks
                assert len(created) == len(set(created))
                for (start, _), represented in outputs.items():
                    block = {
                        (start + d) % L if cyclic else start + d
                        for d in range(block_size)
                    }
                    assert set(represented) == set(range(L)) - block
                    assert len(represented) == L - block_size

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    def test_selected_blocks(self, schedule):
        plan = EnvironmentPlan(10, schedule=schedule)
        blocks = [(8, 2), (2, 2), (5, 2), (2, 2)]
        outputs, _ = execute_symbolically(plan, blocks)
        assert tuple(outputs) == ((2, 2), (5, 2), (8, 2))

    def test_single_target_starts_opposite(self):
        moves = EnvironmentPlan(10).get_moves([(0, 1)], include_deletes=False)
        sites = tuple(site for move in moves for site in move.input_sites)
        assert sites == (5, 6, 4, 7, 3, 8, 2, 9, 1)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            EnvironmentPlan(0)
        with pytest.raises(ValueError):
            EnvironmentPlan(4, schedule="unknown")
        with pytest.raises(ValueError):
            EnvironmentPlan(4).get_moves([(4, 1)])
        with pytest.raises(ValueError):
            EnvironmentPlan(4).get_moves([(0, 0)])
        with pytest.raises(ValueError):
            EnvironmentPlan(4).get_moves([(0, 5)])
        with pytest.raises(ValueError):
            EnvironmentPlan(4, cyclic=False).get_moves([(3, 2)])
        with pytest.raises(ValueError):
            all_blocks(4, 5)

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    @pytest.mark.parametrize("cyclic", [False, True])
    def test_mixed_blocks(self, cyclic, schedule):
        for L in range(1, 10):
            blocks = [
                (start, size)
                for size in range(1, L + 1)
                for start in range(L if cyclic else L - size + 1)
            ]
            plan = EnvironmentPlan(L, cyclic=cyclic, schedule=schedule)
            outputs, _ = execute_symbolically(plan, blocks)
            created = [
                move.output_env
                for move in plan.get_moves(blocks)
                if move.output_env is not None
            ]

            assert set(outputs) == set(blocks)
            assert len(created) == len(set(created))
            for (start, size), represented in outputs.items():
                block = {
                    (start + d) % L if cyclic else start + d
                    for d in range(size)
                }
                assert set(represented) == set(range(L)) - block
                assert len(represented) == L - size

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    def test_mixed_blocks_share_work(self, schedule):
        L = 32
        blocks = [(start, size) for size in (1, 2) for start in range(L)]

        def nconstruct(moves):
            return sum(move.kind in ("init", "contract") for move in moves)

        plan = EnvironmentPlan(L, schedule=schedule)
        together = nconstruct(plan.get_moves(blocks))
        separate = sum(
            nconstruct(plan.get_moves(all_blocks(L, size))) for size in (1, 2)
        )
        assert together < separate

    def test_scaling_and_live_cache(self):
        for L in (8, 16, 32, 64):
            plan = EnvironmentPlan(L)
            moves = plan.get_moves(all_blocks(L, 1))
            ncontract = sum(move.kind == "contract" for move in moves)
            _, peak_cache = execute_symbolically(plan, all_blocks(L, 1))
            assert ncontract <= L * L.bit_length()
            assert peak_cache <= 2 * L.bit_length()
            assert all(
                len(move.input_envs) <= 1
                for move in moves
                if move.kind == "contract"
            )

    def test_open_live_cache(self):
        for L in (8, 16, 32):
            plan = EnvironmentPlan(L, cyclic=False)
            blocks = all_blocks(L, 1, cyclic=False)
            _, peak_cache = execute_symbolically(plan, blocks)
            assert peak_cache <= L

    def test_cut_schedule_scaling(self):
        for L in (8, 16, 32, 64):
            plan = EnvironmentPlan(L, schedule="cut")
            moves = plan.get_moves(all_blocks(L, 1))
            nconstruct = sum(
                move.kind in ("init", "contract") for move in moves
            )
            assert nconstruct <= 3 * L
            assert any(
                len(move.input_envs) == 2
                for move in moves
                if move.kind == "contract"
            )

    @pytest.mark.parametrize("block_size", [1, 2, 5])
    def test_cut_single_start(self, block_size):
        L = 32
        for start in (0, 7, L - block_size, L - 1):
            plan = EnvironmentPlan(L, schedule="cut")
            moves = plan.get_moves([(start, block_size)])
            nsite = sum(len(move.input_sites) for move in moves)
            nmerge = sum(
                move.kind == "contract" and not move.input_sites
                for move in moves
            )
            assert nsite == L - block_size
            assert nmerge <= 1

    def test_cut_selected_starts_and_streaming(self):
        plan = EnvironmentPlan(32, schedule="cut")
        moves = plan.get_moves([(1, 2)], include_deletes=False)
        nconstruct = sum(move.kind in ("init", "contract") for move in moves)
        assert nconstruct == 31

        blocks = [(0, 2), (1, 2)]
        moves = plan.get_moves(blocks, include_deletes=False)
        first_output = next(
            i for i, move in enumerate(moves) if move.kind == "output"
        )
        assert any(
            move.kind in ("init", "contract")
            for move in moves[first_output + 1 :]
        )

    def test_show(self, capsys):
        EnvironmentPlan(4, schedule="cut").show(
            [(0, 1)], include_deletes=False
        )
        printed = capsys.readouterr().out
        assert "init" in printed
        assert "contract" in printed
        assert "output" in printed
        assert " B " in printed

    @pytest.mark.parametrize("schedule", ["tree", "cut"])
    def test_environment_keys_are_intervals(self, schedule):
        plan = EnvironmentPlan(8, schedule=schedule)
        moves = plan.get_moves(all_blocks(8, 2))
        outputs = [
            move.output_env for move in moves if move.output_env is not None
        ]
        assert all(lo < hi for lo, hi in outputs)
        assert len(outputs) == len(set(outputs))

    def test_all_blocks_function(self):
        assert all_blocks(4, 2) == ((0, 2), (1, 2), (2, 2), (3, 2))
        assert all_blocks(4, 2, cyclic=False) == ((0, 2), (1, 2), (2, 2))

    def test_find_1d_block(self):
        assert find_1d_block((0, 3), 6, cyclic=True) == (0, 4)
        assert find_1d_block((1, 4), 6, cyclic=False) == (1, 4)
        with pytest.raises(ValueError):
            find_1d_block((), 6, cyclic=True)
