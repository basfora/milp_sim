"""test things related to School Scenario - 1 and correspondent graph"""
import milp_sim.scenes.make_graph as mg


def test_dimensions():
    """Test if input dimensions to school (dict of room class) are correct w.r.t to floor plan (floorplan.jpg)
    """

    # generate school (dimensions only) -- Gazebo env
    school = mg.ss_2()

    gym = school[1]
    h1 = school[2]
    h2 = school[11]
    h3 = school[12]
    a, b, c = school[3], school[4], school[5]
    d, e, f, g = school[6], school[7], school[8], school[9]
    cafe = school[10]

    assert gym.dim == (17, 31)
    assert h1.dim == (3.4, 12.5)
    assert h2.dim == (50, 3.9)
    assert h3.dim == (10, 8.3)
    assert a.dim == b.dim == c.dim == (5.83, 3.0)
    assert d.dim == e.dim == f.dim == g.dim == (7.90, 5.5)
    assert cafe.dim == (10, 17)

    dx, dy = mg.delta_origin()
    spacex, spacey = mg.space_between()

    assert dx == 53
    assert dy == 0
    assert spacex == [1.5, 1.4, 1.4, 1.4]
    assert spacey == [1.2, 0.6, 0.6]


def test_coordinates():
    school = mg.place_ss2()

    gym = school[1]
    h1 = school[2]
    h2 = school[11]
    h3 = school[12]
    a, b, c = school[3], school[4], school[5]
    d, e, f, g = school[6], school[7], school[8], school[9]
    cafe = school[10]

    assert gym.c == [(53, 0), (53, 31), (70, 31), (70, 0)]
    assert h1.c == [(66.6, 31), (66.6, 43.5), (70.0, 43.5), (70.0, 31)]
    assert a.c == [(60.77, 32.2), (60.77, 35.2), (66.6, 35.2), (66.6, 32.2)]
    assert b.c == [(60.77, 35.8), (60.77, 38.8), (66.6, 38.8), (66.6, 35.8)]
    assert c.c == [(60.77, 39.4), (60.77, 42.4), (66.6, 42.4), (66.6, 39.4)]
    assert h2.c == [(20.0, 43.5), (20.0, 47.4), (70.0, 47.4), (70.0, 43.5)]
    assert d.c == [(60.6, 47.4), (60.6, 52.9), (68.5, 52.9), (68.5, 47.4)]
    assert e.c == [(51.3, 47.4), (51.3, 52.9), (59.2, 52.9), (59.2, 47.4)]
    assert f.c == [(42.0, 47.4), (42.0, 52.9), (49.9, 52.9), (49.9, 47.4)]
    assert g.c == [(32.7, 47.4), (32.7, 52.9), (40.6, 52.9), (40.6, 47.4)]
    assert h3.c == [(10.0, 43.5), (10.0, 51.8), (20.0, 51.8), (20.0, 43.5)]
    assert cafe.c == [(0.0, 33.0), (0.0, 50.0), (10.0, 50.0), (10.0, 33.0)]













