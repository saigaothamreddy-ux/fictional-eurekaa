import numpy as np

from generals.core.grid import Grid, GridFactory


def test_grid_creation():
    map = """
.....
.A##2
...2.
..22.
...B.
    """
    map_nd_array = np.ndarray((5, 5), buffer=np.array(list(map.replace("\n", ""))), dtype="U1")
    grid_str = Grid(map)
    grid_nd_array = Grid(map_nd_array)
    assert grid_str == grid_nd_array

def test_verify_grid():
    map = """
.....
.A##2
...2.
..22.
...B.
    """
    _grid = Grid(map)
    assert Grid.are_generals_connected(_grid.grid)

    map = """
.....
.A##2
##.2.
..###
...B.
    """

    map = Grid.numpify_grid(map)
    assert not Grid.are_generals_connected(map)

    map = """
.....
BA2#2
##.2.
..2##
.....
    """
    map = Grid.numpify_grid(map)
    assert Grid.are_generals_connected(map)

    map = """
...#.
#A2#2
##.#B
..2##
.....
    """
    map = Grid.numpify_grid(map)
    assert not Grid.are_generals_connected(map)

    map = """
...#.
A#2#2
##.#B
..2#.
.....
    """
    map = Grid.numpify_grid(map)
    assert not Grid.are_generals_connected(map)

def test_grid_factory():
    generator = GridFactory()
    generator.rng = np.random.default_rng()
    for _ in range(10):
        grid = generator.generate()
        assert Grid.are_generals_connected(grid.grid)
        height, width = grid.grid.shape
        assert Grid.generals_distance(grid) >= max(height, width) // 2



def test_numpify_map():
    map_str = """
.....
.A##2
...2.
..22.
...B.
    """
    map = Grid.numpify_grid(map_str)
    reference_map = np.array(
        [
            [".", ".", ".", ".", "."],
            [".", "A", "#", "#", "2"],
            [".", ".", ".", "2", "."],
            [".", ".", "2", "2", "."],
            [".", ".", ".", "B", "."],
        ]
    )
    assert (map == reference_map).all()


def test_stringify_map():
    # make map different than from previous example
    np_map = np.array(
        [
            ["#", ".", ".", ".", "A"],
            [".", ".", "#", "#", "2"],
            [".", ".", ".", "2", "."],
            ["4", ".", "2", "2", "."],
            [".", "1", "#", "B", "#"],
        ]
    )

    map_str = Grid.stringify_grid(np_map)
    reference_map = """
#...A
..##2
...2.
4.22.
.1#B#
    """
    assert map_str == reference_map.strip()
