-- ced_obs: 结构化观测导出（Luanti / Craftium 分支用）
--
-- 由仓库根目录的 ced_obs.lua 派生，差异只在"怎么找玩家/写到哪里/数值格式"：
--   * 玩家改为 get_connected_players()[1]（Craftium 的 agent 是唯一的真实连接玩家）
--   * 写到 core.get_worldpath()/ced_obs.txt（每个环境有独立 run_dir，不会串线）
--   * 每个数值按 %.4f 格式化后拼接 —— 默认数字转字符串最多 14 位有效数字，
--     2568 个数会让一帧从 18 KB 涨到 36 KB
--   * dist_h 改为真正的水平距离（原参考实现把三维距离填了进去）
-- schema 的维度、偏移量、帧头格式与 docs/protocol.md 完全一致，未做任何改动。
--
-- 每 tick 一次文件写 + rename，原子替换，Python 侧不会读到半写文件。

local RAY_GRID = 16
local RAY_DIM = 9
local SELF_DIM = 48
local N_ENT = 16
local ENT_DIM = 12
local N_AUDIO = 4
local AUDIO_DIM = 3
local TASK_DIM = 12
local OBS_DIM = RAY_GRID * RAY_GRID * RAY_DIM + SELF_DIM
             + N_ENT * ENT_DIM + N_AUDIO * AUDIO_DIM + TASK_DIM

-- 视野：16:9。水平 70°（半角 35°），垂直按 9/16 收窄到约 43.6°。
-- 之前两个轴都用 35°，是 70°×70° 的正方形，上下被拉宽了近一倍。
local FOV_H_TAN = math.tan(math.rad(35))
local FOV_V_TAN = FOV_H_TAN * 9 / 16
local MAX_DIST = 32

-- 方块类：先用显式表 + 名字哈希兜底，正式版换模糊聚类
local CLASS = {
    air = 0, cave_air = 0,
    ["default:stone"] = 1, ["default:cobble"] = 1, ["default:gravel"] = 1,
    ["default:dirt"] = 2, ["default:dirt_with_grass"] = 3,
    ["default:sand"] = 4, ["default:tree"] = 5, ["default:leaves"] = 6,
    ["default:water_source"] = 7, ["default:lava_source"] = 8,
    ["default:glass"] = 9, ["default:stone_with_iron"] = 10,
    ["default:stone_with_coal"] = 11, ["default:chest"] = 12,
}

local class_of_cache = {}

local function hash(s)
    local h = 5381
    for i = 1, #s do h = (h * 33 + s:byte(i)) % 2147483647 end
    return h
end

local function class_of(name)
    local hit = class_of_cache[name]
    if hit ~= nil then return hit end
    local c = CLASS[name] or (1 + (hash(name) % 190))
    class_of_cache[name] = c
    return c
end

-- 形状类：0 全方块 1 半砖 2 楼梯 3 栅栏 4 玻璃 5 植物 6 流体 7 其他
local function shape_of(name)
    if name:find("slab") then return 1 end
    if name:find("stair") then return 2 end
    if name:find("fence") or name:find("wall") then return 3 end
    if name:find("glass") then return 4 end
    if name:find("flower") or name:find("grass") or name:find("sapling") then return 5 end
    if name:find("water") or name:find("lava") then return 6 end
    return 0
end

local function normalize(v)
    local l = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if l < 1e-6 then return {x = 0, y = 0, z = 0} end
    return {x = v.x / l, y = v.y / l, z = v.z / l}
end

local function cross(a, b)
    return {x = a.y * b.z - a.z * b.y, y = a.z * b.x - a.x * b.z, z = a.x * b.y - a.y * b.x}
end

local function ray_block(player, dir, eyewear)
    local pos = player:get_pos()
    pos.y = pos.y + eyewear
    local target = {x = pos.x + dir.x * MAX_DIST,
                    y = pos.y + dir.y * MAX_DIST,
                    z = pos.z + dir.z * MAX_DIST}
    for pt in core.raycast(pos, target, false, true) do
        if pt.type == "node" then
            local node = core.get_node_or_nil(pt.under)
            if not node then return 0, 0, 0, 0, 0, 0, 0, 0 end
            local def = core.registered_nodes[node.name]
            -- 可穿过的（空气、花草）不终止射线；实心方块与流体终止
            local pass_through = def and def.walkable == false
                                 and (def.liquidtype == nil or def.liquidtype == "none")
            if not pass_through then
                local ip = pt.intersection_point
                local n = pt.intersection_normal
                local dx, dy_, dz = ip.x - pos.x, ip.y - pos.y, ip.z - pos.z
                local dist = math.sqrt(dx * dx + dy_ * dy_ + dz * dz)
                local dist_h = math.sqrt(dx * dx + dz * dz)
                -- 光照取命中面外侧（空气一侧）：命中方块内部的光恒为 0，
                -- 对模型没有信息量；面向观察者的那一侧才是"看上去多亮"。
                local lp = {x = ip.x + n.x * 0.5, y = ip.y + n.y * 0.5, z = ip.z + n.z * 0.5}
                local light = (core.get_node_light(lp) or 0) / 15
                return class_of(node.name), n.x, n.y, n.z, dist, dy_, dist_h, light,
                       shape_of(node.name)
            end
        end
    end
    return 0, 0, 0, 0, 0, 0, 0, 0, 0
end

local function build_obs(player)
    local out = {}
    for i = 1, OBS_DIM do out[i] = 0 end

    local look = player:get_look_dir()
    local up = {x = 0, y = 1, z = 0}
    local right = normalize(cross(look, up))
    local cam_up = cross(right, look)
    local eyewear = 1.5

    local idx = 1
    for v = 0, RAY_GRID - 1 do
        for u = 0, RAY_GRID - 1 do
            local su = (u + 0.5) / RAY_GRID * 2 - 1
            local sv = 1 - (v + 0.5) / RAY_GRID * 2
            local tx, ty = FOV_H_TAN * su, FOV_V_TAN * sv
            local dir = normalize({
                x = look.x + tx * right.x + ty * cam_up.x,
                y = look.y + tx * right.y + ty * cam_up.y,
                z = look.z + tx * right.z + ty * cam_up.z,
            })
            local c, nx, ny, nz, d, dy_, dh, light, shape = ray_block(player, dir, eyewear)
            out[idx + 0] = c
            out[idx + 1] = nx
            out[idx + 2] = ny
            out[idx + 3] = nz
            out[idx + 4] = math.min(d, MAX_DIST) / MAX_DIST
            out[idx + 5] = math.max(-1, math.min(1, dy_ / MAX_DIST))
            out[idx + 6] = math.min(dh, MAX_DIST) / MAX_DIST
            out[idx + 7] = light
            out[idx + 8] = shape
            idx = idx + 9
        end
    end

    -- self 段（48）
    local p = player:get_pos()
    local yaw = player:get_look_horizontal()
    local pitch = player:get_look_vertical()
    out[idx + 0] = p.x / 32
    out[idx + 1] = p.y / 32
    out[idx + 2] = p.z / 32
    out[idx + 3] = math.sin(yaw)
    out[idx + 4] = math.cos(yaw)
    out[idx + 5] = math.sin(pitch)
    out[idx + 6] = math.cos(pitch)

    -- 速度：服务端同步的玩家速度（之前恒为 0，模型完全不知道自己在动）
    local vel = {x = 0, y = 0, z = 0}
    local ok, v = pcall(player.get_player_velocity, player)
    if not ok or type(v) ~= "table" then
        ok, v = pcall(player.get_velocity, player)
    end
    if ok and type(v) == "table" then vel = v end
    out[idx + 7] = vel.x / 32
    out[idx + 8] = vel.y / 32
    out[idx + 9] = vel.z / 32

    -- on_ground：脚下 0.1 格是实心方块
    local below = core.get_node_or_nil({x = p.x, y = p.y - 0.1, z = p.z})
    local bdef = below and core.registered_nodes[below.name]
    out[idx + 10] = (bdef and bdef.walkable) and 1 or 0

    -- 潜行 / 冲刺：服务端能拿到玩家的控制位
    local okc, ctrl = pcall(player.get_player_control, player)
    if okc and type(ctrl) == "table" then
        out[idx + 11] = ctrl.sneak and 1 or 0
        out[idx + 12] = ctrl.aux1 and 1 or 0
    end

    local okh, hp = pcall(player.get_hp, player)
    out[idx + 13] = (okh and tonumber(hp)) and hp / 20 or 0
    -- 饥饿：本环境（minetest_game）没有饥饿系统，保持 0，不假装是满的
    out[idx + 14] = 0

    local okw, wi = pcall(player.get_wield_index, player)
    out[idx + 16] = (okw and wi) and (wi - 1) / 8 or 0

    -- 眼位光照：之前硬编码 1（等于满光），现在读真实值
    local eye_light = core.get_node_light({x = p.x, y = p.y + 1.5, z = p.z})
    out[idx + 17] = (eye_light or 0) / 15
    -- ents / audio 段留 0，task 段由 fill_task 填
    return out
end

-- 帧格式见 docs/protocol.md：第一行 7 个字段，第二行 OBS_DIM 个数值
local SCHEMA = "obs-v0.1"
local TICK = 0
local EPISODE = 0
local REWARD, TERM, TRUNC = 0, 0, 0

-- ---------------------------------------------------------------- 任务：go_to_point
-- 目标点由 fixed_map_seed 决定，相对出生点 10–20 格；距离奖励 + 到达终止。
-- 这块是 L1 的第一个任务，用来把"观测 → 教师 → 分片 → 训练"整条链路跑在有物理的世界里。
local TASK_ID = 1
local REACH = 0.8
local GOAL_MIN = tonumber(core.settings:get("ced_goal_min") or "10") or 10
local GOAL_MAX = tonumber(core.settings:get("ced_goal_max") or "20") or 20
-- 房间边界（room 环境实测）：目标点必须落在里面，否则教师够不着，
-- 之前就是没限制，5 个失败的 episode 目标全在墙外。
local ROOM_XMIN = tonumber(core.settings:get("ced_room_xmin") or "-13.2") or -13.2
local ROOM_XMAX = tonumber(core.settings:get("ced_room_xmax") or "4.2") or 4.2
local ROOM_ZMIN = tonumber(core.settings:get("ced_room_zmin") or "-24.2") or -24.2
local ROOM_ZMAX = tonumber(core.settings:get("ced_room_zmax") or "-0.8") or -0.8
local MARGIN = REACH + 0.4
local GOAL = nil
local PREV_DIST = nil
local INIT_DIST = nil
local TASK_DONE = false

-- 目标点所在的两格必须是空气（脚 + 头）
local function clear_at(gx, gz, y0)
    for dy = 0, 1 do
        local n = core.get_node_or_nil({x = gx, y = y0 + dy, z = gz})
        local def = n and core.registered_nodes[n.name]
        if not def or def.walkable then return false end
    end
    return true
end

local function init_task(player)
    local p = player:get_pos()
    local seed = tonumber(core.settings:get("fixed_map_seed") or "0") or 0
    local s = (seed * 1103515245 + 12345) % 2147483648
    local function rnd()
        s = (s * 1103515245 + 12345) % 2147483648
        return s / 2147483648
    end
    local y0 = math.floor(p.y) + 1
    local best = nil
    for _ = 1, 64 do
        local ang = rnd() * 2 * math.pi
        local d = GOAL_MIN + rnd() * (GOAL_MAX - GOAL_MIN)
        local gx, gz = p.x + math.cos(ang) * d, p.z + math.sin(ang) * d
        if gx >= ROOM_XMIN + MARGIN and gx <= ROOM_XMAX - MARGIN
                and gz >= ROOM_ZMIN + MARGIN and gz <= ROOM_ZMAX - MARGIN
                and clear_at(math.floor(gx) + 0.5, math.floor(gz) + 0.5, y0) then
            best = {x = gx, z = gz}
            break
        end
    end
    if not best then
        local cx = math.max(ROOM_XMIN + MARGIN, math.min(ROOM_XMAX - MARGIN, p.x))
        local cz = math.max(ROOM_ZMIN + MARGIN, math.min(ROOM_ZMAX - MARGIN, p.z - GOAL_MIN))
        best = {x = cx, z = cz}
    end
    GOAL = best
    PREV_DIST = math.sqrt((GOAL.x - p.x) ^ 2 + (GOAL.z - p.z) ^ 2)
    INIT_DIST = PREV_DIST
end

local function update_task(player)
    if not GOAL then return end
    local p = player:get_pos()
    local dx, dz = GOAL.x - p.x, GOAL.z - p.z
    local d = math.sqrt(dx * dx + dz * dz)
    if PREV_DIST then REWARD = (PREV_DIST - d) end
    PREV_DIST = d
    if d < REACH and not TASK_DONE then
        TASK_DONE = true
        TERM = 1
        REWARD = REWARD + 10
    end
end

-- 任务段的局部偏移：OBS_DIM - TASK_DIM（0 基），out 表是 1 基
local OFF_TASK_LOCAL = OBS_DIM - TASK_DIM

local function fill_task(out, p)
    if not GOAL then return end
    local dx, dz = GOAL.x - p.x, GOAL.z - p.z
    local d = math.sqrt(dx * dx + dz * dz)
    out[OFF_TASK_LOCAL + 1] = TASK_ID
    out[OFF_TASK_LOCAL + 2] = dx / 32
    out[OFF_TASK_LOCAL + 3] = dz / 32
    out[OFF_TASK_LOCAL + 4] = math.min(d, 32) / 32
    out[OFF_TASK_LOCAL + 5] = INIT_DIST and math.max(0, math.min(1, 1 - d / INIT_DIST)) or 0
end

-- 任务脚本通过这些函数写奖励与终止；不要在 Python 侧重算
function ced_obs_set_reward(r) REWARD = r end
function ced_obs_set_termination() TERM = 1 end
function ced_obs_set_truncation() TRUNC = 1 end
function ced_obs_new_episode() EPISODE = EPISODE + 1; REWARD, TERM, TRUNC = 0, 0, 0 end

local function write_obs(path, values)
    local parts = {}
    for i = 1, #values do parts[i] = string.format("%.4f", values[i]) end
    local head = string.format("%s %d %d %d %.6f %d %d",
        SCHEMA, TICK, EPISODE, OBS_DIM, REWARD, TERM, TRUNC)
    local tmp = path .. ".tmp"
    local f = io.open(tmp, "w")
    if not f then return end
    f:write(head .. "\n" .. table.concat(parts, " ") .. "\n")
    f:close()
    os.rename(tmp, path)                       -- 原子替换，Python 不会读到半写文件
end

local TICK_BUDGET_MS = 0
local TICK_MAX_MS = 0
local LOOK_FILE = "look.txt"     -- 每行：<yaw_rad> <pitch_rad>，Python 写，Lua 每 tick 应用

-- 服务端的 look 同步不可靠（pitch 完全不跟进），所以视线由 Python 侧权威指定。
-- 只在目标值变化时调用 set_look_*：每次调用都会向客户端发一次 MovePlayer，
-- 客户端按增量叠加，每 tick 都调用会让它的 yaw 无限累加（实测 60 tick → 5156°）。
local last_yaw, last_pitch = nil, nil

local function apply_look(player)
    local f = io.open(core.get_worldpath() .. "/" .. LOOK_FILE, "r")
    if not f then return end
    local line = f:read("*l")
    f:close()
    if not line then return end
    local yaw, pitch = line:match("([%-%d%.eE]+)%s+([%-%d%.eE]+)")
    if not yaw then return end
    yaw, pitch = tonumber(yaw), tonumber(pitch)
    if last_yaw == nil or math.abs(yaw - last_yaw) > 1e-4
            or math.abs(pitch - last_pitch) > 1e-4 then
        player:set_look_horizontal(yaw)
        player:set_look_vertical(pitch)
        last_yaw, last_pitch = yaw, pitch
    end
end

core.register_globalstep(function(_)
    local player = core.get_connected_players()[1]
    if not player then return end
    local t0 = core.get_us_time()
    TICK = TICK + 1
    if not GOAL then init_task(player) end
    apply_look(player)
    update_task(player)
    local values = build_obs(player)
    fill_task(values, player:get_pos())
    local path = core.get_worldpath() .. "/ced_obs.txt"
    write_obs(path, values)
    local dt = (core.get_us_time() - t0) / 1000
    TICK_BUDGET_MS = TICK_BUDGET_MS * 0.98 + dt * 0.02
    if dt > TICK_MAX_MS then TICK_MAX_MS = dt end
    if TICK % 20 == 0 then
        -- 每 200 tick 写一次，用来判断射线是否拖慢 tick（EXECUTION S7 的 5ms 门禁）
        local sf = io.open(core.get_worldpath() .. "/ced_obs_stats.txt", "w")
        if sf then
            local p = player:get_pos()
            sf:write(string.format(
                "tick %d avg_ms %.3f max_ms %.3f yaw %.2f pitch %.2f pos %.2f %.2f %.2f\n",
                TICK, TICK_BUDGET_MS, TICK_MAX_MS,
                player:get_look_horizontal(), player:get_look_vertical(), p.x, p.y, p.z))
            sf:close()
        end
    end
end)

-- TODO(day-6 之后)：动作通道。Lua 每 tick 读 act.txt 的 `<tick> <14 ints>`，
-- 用 core.set_player_controls 之类的接口施加，并在应用后记录 last_applied。
-- 接口约定见 docs/protocol.md；Craftium 目前的动作从它自己的通道进来，
-- 两条路不能同时生效，切换时要显式禁用 Craftium 的控制。
