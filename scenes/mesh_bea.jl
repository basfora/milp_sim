
using Makie
using StaticArrays
using Printf


room_count = [0]  # global
v = Vector{Int64}()  # global

struct NODE
    dir_pair::String
    x::Union{Nothing,Float64}
    y::Union{Nothing,Float64}
    function NODE(; x=nothing, y=nothing, dir_pair::String)
        return new(dir_pair, x, y)
    end
end
is_x(n::NODE) = !isnothing(n.x)
is_y(n::NODE) = !isnothing(n.y)

###

mutable struct Attached_Type
    room
    dir::String
    Δ_pos::Float64
    door_pos_in_child::Float64
    function Attached_Type(room, dir::String, Δ_pos::Float64, door_pos_in_child::Float64)
        return new(room, dir, Δ_pos, door_pos_in_child)
    end
end

###

mutable struct Room
    id::Int64
    x::Float64
    y::Float64
    refine_method
    name::String
    attached::Vector{Attached_Type}
    nodes::Vector{NODE}
    scene_element
    is_active::Bool
    global_bounds::NTuple{4,Float64}
    is_pos_x::Bool
    is_pos_y::Bool
    function Room(x, y, refine_method, name::String)
        attached = Vector{Attached_Type}()
        nodes = Vector{NODE}()
        room_count[1] += 1
        return new(room_count[1], x, y, refine_method, name, attached, nodes, nothing, false)
    end
end

get_Δx(r::Room) = r.global_bounds[2] -  r.global_bounds[1]
get_Δy(r::Room) = r.global_bounds[4] -  r.global_bounds[3]
get_x0(r::Room) = r.global_bounds[1]
get_x1(r::Room) = r.global_bounds[2]
get_y0(r::Room) = r.global_bounds[3]
get_y1(r::Room) = r.global_bounds[4]

get_nx(r::Room) = get_nx(r.refine_method)
get_ny(r::Room) = get_ny(r.refine_method)

function search_for_x_constraint(r::Room)
    nodes = r.nodes
    for node_k = nodes
        is_x(node_k) && (return node_k.x)
    end
    return nothing
end
function search_for_y_constraint(r::Room)
    nodes = r.nodes
    for node_k = nodes
        is_y(node_k) && (return node_k.y)
    end
    return nothing
end

is_con_y(r::Room) = !isnothing(search_for_y_constraint(r))
is_con_x(r::Room) = !isnothing(search_for_x_constraint(r))

###

abstract type RefineType end

mutable struct RefineHoriz <: RefineType
    n::Int64
    function RefineHoriz()
        return new(1)
    end
end
mutable struct RefineVert <: RefineType
    n::Int64
    function RefineVert()
        return new(1)
    end
end
mutable struct RefineRectUR <: RefineType
    nx::Int64
    ny::Int64
    function RefineRectUR()
        return new(1, 1)
    end
end
mutable struct RefineRectDR <: RefineType
    nx::Int64
    ny::Int64
    function RefineRectDR()
        return new(1, 1)
    end
end
mutable struct RefineTri <: RefineType end

refine_x!(r::Union{RefineRectUR,RefineRectDR}) = (r.nx += 1; return nothing)
refine_y!(r::Union{RefineRectUR,RefineRectDR}) = (r.ny += 1; return nothing)
refine_x!(r::Union{RefineHoriz,RefineVert})    = (r.n += 1;  return nothing)
refine_y!(r::Union{RefineHoriz,RefineVert})    = (r.n += 1;  return nothing)

get_nx(r::Union{RefineRectUR,RefineRectDR}) = r.nx
get_ny(r::Union{RefineRectUR,RefineRectDR}) = r.ny

get_nx(r::RefineHoriz) = r.n
get_ny(r::RefineHoriz) = 1

get_nx(r::RefineVert) = 1
get_ny(r::RefineVert) = r.n

###

function add_attached!(r::Room, r_new::Room, dir::String, offset::Float64, opening::Float64)
    a = Attached_Type(r_new, dir, offset, opening)
    push!(r.attached, a)
end


plot_room!(r::Room) = plot_room!(r, 0.0, 0.0)

function plot_room!(r::Room, x0::Float64, y0::Float64)
    x1 = r.x + x0
    y1 = r.y + y0
    x = [x0, x1, x1, x0, x0]
    y = [y0, y0, y1, y1, y0]
    plot!(x, y, color="black")
    r.global_bounds = (x0, x1, y0, y1)
    for attached_k = r.attached
        x0_k, y0_k = calc_offset(r, attached_k, x0, y0)
        plot_room!(attached_k.room, x0_k, y0_k)
    end
end

function calc_offset(r::Room, a::Attached_Type, x0::Float64, y0::Float64)
    if a.dir == "r"
        x0 += r.x
        y0 += a.Δ_pos
        door_pos = y0 + a.door_pos_in_child
        push!(r.nodes,      NODE(y=door_pos, dir_pair="r"))
        push!(a.room.nodes, NODE(y=door_pos, dir_pair="l"))
    elseif a.dir == "u"
        y0 += r.y
        x0 += a.Δ_pos
        door_pos = x0 + a.door_pos_in_child
        push!(r.nodes,      NODE(x=door_pos, dir_pair="u"))
        push!(a.room.nodes, NODE(x=door_pos, dir_pair="d"))
    elseif a.dir == "d"
        y0 -= a.room.y
        x0 += a.Δ_pos
        door_pos = x0 + a.door_pos_in_child
        push!(r.nodes,      NODE(x=door_pos, dir_pair="d"))
        push!(a.room.nodes, NODE(x=door_pos, dir_pair="u"))
    elseif a.dir == "l"
        x0 -= a.room.x
        y0 += a.Δ_pos
        door_pos = y0 + a.door_pos_in_child
        push!(r.nodes,      NODE(y=door_pos, dir_pair="l"))
        push!(a.room.nodes, NODE(y=door_pos, dir_pair="r"))
    else
        error("not implemented error")
    end

    return x0, y0
end

function do_start_smart(v_con::Nothing, r::Room, n::Int64, is_x::Bool)
    V0 = ifelse(is_x, get_x0, get_y0)(r)
    V1 = ifelse(is_x, get_x1, get_y1)(r)
    v_space = (V1 - V0) / n
    v_start = V0 + v_space / 2
    return v_start, v_space, true
end

function find_con_side(r::Room, v_con::Float64, is_x::Bool)
    v_con = ifelse(is_x, search_for_x_constraint, search_for_y_constraint)(r)
    V0 = ifelse(is_x, get_x0, get_y0)(r)
    V1 = ifelse(is_x, get_x1, get_y1)(r)
    return abs(v_con - V0) > abs(v_con - V1)
end

function do_start_smart(v_con::Float64, r::Room, n::Int64, is_x::Bool)
    (n == 1) && (return v_con, 0.0, true)
    V0 = ifelse(is_x, get_x0, get_y0)(r)
    V1 = ifelse(is_x, get_x1, get_y1)(r)
    is_pos = abs(v_con - V0) < abs(v_con - V1)
    v0 = v_con
    v1 = V1 + (V0 - v0)
    v_space = (v1 - v0) / (n - 1)
    if !is_pos
        v_space *= -1.0
        v0 = v1
    end
    return v0, v_space, is_pos
end

function do_start_smart(r::Room, is_x::Bool)
    n = ifelse(is_x, get_nx, get_ny)(r)
    v_con = ifelse(is_x, search_for_x_constraint, search_for_y_constraint)(r)
    v_start, v_space, is_pos = do_start_smart(v_con, r, n, is_x)
    return v_start, v_space, n, is_pos
end

function plot_node(r::Room, ::RefineTri)
    n1 = r.nodes[1]
    n2 = r.nodes[2]

    is_y(n1) || error("not implemented")
    is_y(n2) || error("not implemented")
    (length(r.nodes) == 2) || error("not implemented")

    X0 = r.global_bounds[1]
    X1 = r.global_bounds[2]
    ΔX = X1 - X0

    x0 = X0 + ΔX * 0.25
    x1 = X0 + ΔX * 0.75

    y0 = n1.y
    y1 = n2.y
    y1_below = y1 - 2*(y1 - y0)

    x = [x0, x1, x1]
    y = [y0, y1, y1_below]

    return x, y, true, true
end

function plot_node(r::Room, ::Union{RefineRectDR,RefineVert,RefineHoriz,RefineRectUR})
    start_x, space_x, nx, is_pos_x = do_start_smart(r, true)
    start_y, space_y, ny, is_pos_y = do_start_smart(r, false)
    x = Vector{Float64}()
    y = Vector{Float64}()
    for ix = 0:(nx-1)
        for iy = 0:(ny-1)
            push!(x, start_x + space_x * ix)
            push!(y, start_y + space_y * iy)
        end
    end
    return x, y, is_pos_x, is_pos_y
end

function plot_node!(r::Room)
    x, y, is_pos_x, is_pos_y = plot_node(r, r.refine_method)
    r.is_pos_x = is_pos_x
    r.is_pos_y = is_pos_y

    # prevent remove points from previous iterations if necessary
    index_in_v = findall(v .== r.id)
    if length(index_in_v) != 0
        index = only(index_in_v)
        delete!(scene, scene[end - index + 1])
        deleteat!(v, index)
    end
    scatter!(x, y)
    pushfirst!(v, r.id)

    for attached_k = r.attached # recurse
        is_pos_x, is_pos_y = plot_node!(attached_k.room)
    end

    return is_pos_x, is_pos_y
end

###

struct Graph
    points::Vector{SVector{2,Float64}}
    edges::Vector{SVector{2,Int64}}
    function Graph()
        points = Vector{SVector{2,Float64}}()
        edges = Vector{SVector{2,Int64}}()
        return new(points, edges)
    end
end

function add_graph_room!(g::Graph, r::Room, ::RefineTri, ee_parent)
    x, y = plot_node(r, RefineTri())

    len_orig = length(g.points)

    for k = 1:3
        push!(g.points, SVector{2,Float64}(x[k], y[k]))
    end

    for k = 1:3
        push!(g.edges, SVector{2,Int64}(k, rem(k, 3) + 1) .+ len_orig)
    end

    return ExtremeInd(1, 1, 3, 2, len_orig)
end

function add_graph_room!(g::Graph, r::Room, ::Union{RefineRectDR,RefineVert,RefineHoriz,RefineRectUR}, ee_parent)
    ind_2_node(ind_y::Int64, ind_x::Int64) = ind_y + (ind_x - 1) * ny

    start_x, space_x, nx, is_reg_x = do_start_smart(r, true)
    start_y, space_y, ny, is_reg_y = do_start_smart(r, false)
    point_matrix = Matrix{SVector{2,Float64}}(undef, ny, nx)
    edge_vec = Vector{SVector{2,Int64}}()
    for ix = 1:nx
        for iy = 1:ny
            x = start_x + space_x * (ix - 1)
            y = start_y + space_y * (iy - 1)
            point_matrix[iy, ix] = SVector{2,Float64}(x, y)
            index = ind_2_node(iy, ix)
            if ix != nx
                index_right = ind_2_node(iy, ix + 1)
                push!(edge_vec, SVector{2,Int64}(index, index_right))
            end
            if iy != ny
                index_down = ind_2_node(iy + 1, ix)
                push!(edge_vec, SVector{2,Int64}(index, index_down))
            end
        end
    end

    len_orig = length(g.points)

    for point_k = point_matrix
        # point_k = SVector{2,Float64}(point_k[1] + randn() * 0.15, point_k[2] + randn() * 0.15)
        push!(g.points, point_k)
    end

    for edge_k = edge_vec
        push!(g.edges, edge_k .+ len_orig)
    end

    return ExtremeInd(ind_2_node(ny, 1), ind_2_node(1, 1), ind_2_node(ny, nx), ind_2_node(1, nx), len_orig)
end

struct ExtremeInd
    i_ul::Int64
    i_dl::Int64
    i_ur::Int64
    i_dr::Int64
    function ExtremeInd(i_ul, i_dl, i_ur, i_dr, len_orig)
        return new(i_ul + len_orig, i_dl + len_orig, i_ur + len_orig, i_dr + len_orig)
    end
end

function add_graph!(g::Graph, r::Room, a=nothing)
    ee_parent = add_graph_room!(g, r, r.refine_method, a)

    for attached_k = r.attached
        ee_child = add_graph!(g, attached_k.room, attached_k)

        if attached_k.dir == "r"
            if r.is_pos_y
                i_parent = ee_parent.i_dr
            else
                i_parent = ee_parent.i_ur
            end
            i_child = ee_child.i_ul
            push!(g.edges, SVector{2,Int64}(i_parent, i_child))
        elseif attached_k.dir == "u"
            v_con = search_for_x_constraint(attached_k.room)
            i_child = ifelse(find_con_side(attached_k.room, v_con, true), ee_child.i_dr, ee_child.i_dl)
            i_parent = ifelse(find_con_side(r, v_con, true), ee_parent.i_ur, ee_parent.i_ul)
            push!(g.edges, SVector{2,Int64}(i_parent, i_child))
        elseif attached_k.dir == "d"
            v_con = search_for_x_constraint(attached_k.room)
            i_child = ifelse(find_con_side(attached_k.room, v_con, true), ee_child.i_ur, ee_child.i_ul)
            i_parent = ifelse(find_con_side(r, v_con, true), ee_parent.i_dr, ee_parent.i_dl)
            push!(g.edges, SVector{2,Int64}(i_parent, i_child))
        elseif attached_k.dir == "l"
            v_con = search_for_y_constraint(attached_k.room)
            i_child = ifelse(find_con_side(attached_k.room, v_con, false), ee_child.i_ur, ee_child.i_dr)
            i_parent = ifelse(find_con_side(r, v_con, false), ee_parent.i_ul, ee_parent.i_dl)
            push!(g.edges, SVector{2,Int64}(i_parent, i_child))
        end

    end

    return ee_parent
end

function print_graph(g::Graph)
    println("===================")
    println()

    dx = -6.95
    dy = 31.8

    # translate graph to my coordinates
    i = 1
    for point_k = g.points
        s0 = @sprintf("%d", i)
        s1 = @sprintf("%0.5f", point_k[1] + dx)
        s2 = @sprintf("%0.5f", point_k[2] + dy)
        println(s0," ", s1, " ", s2)
        i += 1
    end
    println()
    j = 1
    for edge_k = g.edges
        println(j," ",edge_k[1], " ", edge_k[2])
        j += 1
    end
    println()
end

function plot_graph(g::Graph)

    # # prevent remove points from previous iterations if necessary
    # index_in_v = findall(v .== -9999)
    # if length(index_in_v) != 0
    #     index = only(index_in_v)
    #     delete!(scene, scene[end - index + 1])
    #     deleteat!(v, index)
    # end

    for edge_k = g.edges
        p1 = g.points[edge_k[1]]
        p2 = g.points[edge_k[2]]
        plot!([p1[1], p2[1]], [p1[2], p2[2]])
    end

    # pushfirst!(v, -9999)
end

###

function create_layout_gazebo(is_debug::Bool=false)
    width = 3.9
    wid2 = 1.95

    # cafe
    r1 = Room(10.0, 17.0, RefineRectUR(), "room_1")

    is_debug && (r1.refine_method.nx = 3)
    is_debug && (r1.refine_method.ny = 5)

    # H3
    r2 = Room(10.0, 8.3, RefineTri(), "room_2")
    add_attached!(r1, r2, "r", 10.5, 3.3)

    # H2 - Stairs
    r3 = Room(12.0, width, RefineHoriz(), "room_3")
    add_attached!(r2, r3, "r", 0.0, wid2)

    is_debug && (r3.refine_method.n = 4)

    r_last_hall = r3

    for k = 1:4

        hx = ifelse(k == 4, 9.4, 9.3)
        r4 = Room(hx, width, RefineHoriz(), "room_4")
        add_attached!(r_last_hall, r4, "r", 0.0, wid2)

        r5 = Room(7.9, 5.5, RefineRectDR(), "room_5")
        add_attached!(r4, r5, "u", 0.0, 7.45)

        r_last_hall = r4

        is_debug && (r_last_hall.refine_method.n = 2)
        is_debug && (r5.refine_method.nx = 2)
        is_debug && (r5.refine_method.ny = 2)
    end

    width = 3.4
    for k = 1:3

        h = [4.1, 3.6, 4.8]
        wd = 1.5

        # H1
        r6 = Room(width, h[k], RefineVert(), "room_6")

        Δx = ifelse(k == 1, 6.0, 0.0)
        add_attached!(r_last_hall, r6, "d", Δx, wd)

        is_debug && (r6.refine_method.n = 2)

        # rooms A-B-C
        r7 = Room(5.83, 3.0, RefineHoriz(), "room_7")

        dh = ifelse(k==3, 1.2, 0.0)
        add_attached!(r6, r7, "l", dh, 1.5)

        is_debug && (r7.refine_method.n = 3)

        r_last_hall = r6
    end

    w8 = 17.0
    r8 = Room(w8, 31.0, RefineRectUR(), "room_8")
    add_attached!(r_last_hall, r8, "d", - 13.6, 15.8)

    is_debug && (r8.refine_method.nx = 3)
    is_debug && (r8.refine_method.ny = 5)

    return r1
end

function create_layout_bloat(is_debug::Bool=false)

    width = 4.29
    wid2 = width/2

    # cafe dimensions: (12.65, 23.2)
    r1 = Room(12.65, 23.2, RefineRectUR(), "room_1")

    is_debug && (r1.refine_method.nx = 3)
    is_debug && (r1.refine_method.ny = 5)

    # Hall 3 dimensions (11.0, 9.13)
    r2 = Room(11.0, 9.13, RefineTri(), "room_2")
    # Attach cafe-h3:  16.05, 2.4
    add_attached!(r1, r2, "r", 16.05, wid2)

    # H2.S dimensions (8.58, 4.29)
    r3 = Room(8.58, width, RefineHoriz(), "room_3")
    # H2.S attach  (0.0, 2.145)
    add_attached!(r2, r3, "r", 0.0, wid2)

    is_debug && (r3.refine_method.n = 4)

    r_last_hall = r3

    # rooms G-F-E-D (11.99, 4.29) (10.45, 4.29)
    wh2 = 4.29
    wh2h = wh2/2
    for k = 1:4

        # H2.D dimensions <> H2.G, H2.F, H2.E dimensions
        hx = ifelse(k == 4, 10.45, 11.99)
        r4 = Room(hx, wh2, RefineHoriz(), "room_4")
        # Attach classrooms: (0, 2.145)
        add_attached!(r_last_hall, r4, "r", 0.0, wh2h)

        # G-D rooms dimensions (10.45, 6.05)
        r5 = Room(10.45, 6.05, RefineRectDR(), "room_5")
        # G-D attach doors (0.0, 9.2)
        add_attached!(r4, r5, "u", 0.0, 9.2)

        r_last_hall = r4

        is_debug && (r_last_hall.refine_method.n = 2)
        is_debug && (r5.refine_method.nx = 2)
        is_debug && (r5.refine_method.ny = 2)
    end

    wh1 = 3.74
    wh1h = wh1/2

    for k = 1:3

        # H1C (3.74, 4.51) , H1B (3.74, 3.96), H1A (3.74, 5.28)
        h = [4.51, 3.96, 5.28]
        r6 = Room(width, h[k], RefineVert(), "room_6")

        # Attach H1-H2 (6.71, 1.87)
        Δx = ifelse(k == 1, 6.71, 0.0)
        add_attached!(r_last_hall, r6, "d", Δx, wh1h)

        is_debug && (r6.refine_method.n = 2)

        # Rooms A-B-C dimensions (6.41, 3.3)
        r7 = Room(6.41, 3.3, RefineHoriz(), "room_7")

        # H1B-C attach (0.0, 1.8), H1-A attach (1.32, 1.8)
        dh = ifelse(k==3, 1.32, 0.0)
        add_attached!(r6, r7, "l", dh, 1.8)

        is_debug && (r7.refine_method.n = 3)

        r_last_hall = r6
    end

    w8 = 18.7
    # Gym dimensions (18.7, 34.1)
    r8 = Room(w8, 34.1, RefineRectUR(), "room_8")
    # Attach Gym (-14.96, 16.83)
    add_attached!(r_last_hall, r8, "d", - 14.96, 16.83)

    is_debug && (r8.refine_method.nx = 3)
    is_debug && (r8.refine_method.ny = 5)

    return r1
end

scene = Scene()
r1 = create_layout_bloat()
plot_room!(r1)
plot_node!(r1)

### Event-based stuff

function print_active_room(r::Room)
    if r.is_active
        println(r.name, " is active!")
    end

    for attached_k = r.attached
        print_active_room(attached_k.room)
    end
end

function set_all_room_inactive(r::Room)
    r.is_active = false
    for attached_k = r.attached
        set_all_room_inactive(attached_k.room)
    end
end

function set_active_room(r::Room, x::Float32, y::Float32)
    if (r.global_bounds[1] < x < r.global_bounds[2])
        if (r.global_bounds[3] < y < r.global_bounds[4])
            r.is_active = true
        end
    end
    for attached_k = r.attached
        set_active_room(attached_k.room, x, y)
    end
end

function refine_active!(r::Room, fun_name::Function)
    r.is_active && fun_name(r.refine_method)
    for attached_k = r.attached
        refine_active!(attached_k.room, fun_name)
    end
end

display(scene)

xlims!(-40, 75.0)
ylims!(-40, 40.0)

if true
    dir = lift(scene.events.keyboardbuttons) do but
        is_lr = ispressed(but, Keyboard.left) || ispressed(but, Keyboard.right)
        is_ud = ispressed(but, Keyboard.up) || ispressed(but, Keyboard.down)

        if is_lr
            println("in keyboard event loop x")
            refine_active!(r1, refine_x!)
            plot_node!(r1)
        end

        if is_ud
            println("in keyboard event loop y")
            refine_active!(r1, refine_y!)
            plot_node!(r1)
        end

        if ispressed(but, Keyboard.space)
            g = Graph()
            add_graph!(g, r1)
            plot_graph(g)
            print_graph(g)
        end
        return 0
    end

    pos = lift(scene.events.mouseposition) do mpos
        # println("HIT")

        x0, x1 = mouseposition(scene)

        set_all_room_inactive(r1)
        set_active_room(r1, x0, x1)
        return 0
    end
end


println()
println()
println("DONE!!!")
