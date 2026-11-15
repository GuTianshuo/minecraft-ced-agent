-- ced_obs: 结构化观测导出（Luanti / Craftium 分支用）
--
-- 每 tick 从假人眼位发 16x16 条射线，按 schema_v0.py 的布局写成一行浮点数。
-- Python 侧用 StructuredObsChannel 读取（envs.py）。
--
-- 状态：参考实现，未在真机验证。第 6 天的 spike 任务就是让它跑通。
-- 已知取舍：
--   * 文本格式（20 KB/s/环境）而不是二进制 —— Luanti 常跑 LuaJIT，没有 string.pack
--   * 每 tick 一次文件写 + rename，2 KB 级别，先不管开销，实测再说
--   * 方块类目前是"名字哈希到 200 类"的占位，正式版要按 design.md 的模糊聚类重写

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

local FOV_HALF = math.rad(35)
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

local function ray_block(player, dir)
    local pos = player:get_pos()
    pos.y = pos.y + 1.5                       -- 眼位；用 get_properties 的 eye_height 替换
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
                local dist = math.sqrt((ip.x - pos.x) ^ 2 + (ip.y - pos.y) ^ 2 + (ip.z - pos.z) ^ 2)
                local light = (core.get_node_light(pt.under) or 0) / 15
                return class_of(node.name), n.x, n.y, n.z, dist, ip.y - pos.y, light, shape_of(node.name)
            end
        end
    end
    return 0, 0, 0, 0, 0, 0, 0, 0
end

local function build_obs(player)
    local out = {}
    for i = 1, OBS_DIM do out[i] = 0 end

    local look = player:get_look_dir()
    local up = {x = 0, y = 1, z = 0}
    local right = normalize(cross(look, up))
    local cam_up = cross(right, look)

    local idx = 1
    for v = 0, RAY_GRID - 1 do
        for u = 0, RAY_GRID - 1 do
            local su = (u + 0.5) / RAY_GRID * 2 - 1
            local sv = 1 - (v + 0.5) / RAY_GRID * 2
            local tx, ty = math.tan(FOV_HALF) * su, math.tan(FOV_HALF) * sv
            local dir = normalize({
                x = look.x + tx * right.x + ty * cam_up.x,
                y = look.y + tx * right.y + ty * cam_up.y,
                z = look.z + tx * right.z + ty * cam_up.z,
            })
            local c, nx, ny, nz, d, dy, light, shape = ray_block(player, dir)
            out[idx + 0] = c
            out[idx + 1] = nx
            out[idx + 2] = ny
            out[idx + 3] = nz
            out[idx + 4] = math.min(d, MAX_DIST) / MAX_DIST
            out[idx + 5] = math.max(-1, math.min(1, dy / MAX_DIST))
            out[idx + 6] = math.min(d, MAX_DIST) / MAX_DIST
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
    out[idx + 10] = 1                          -- TODO: 真实 on_ground
    out[idx + 13] = 1                          -- TODO: health/20
    out[idx + 14] = 1                          -- TODO: hunger/20
    out[idx + 17] = 1                          -- TODO: 眼位光照/15
    -- ents / audio / task 段留 0，由任务脚本填（见 TODO）
    return out
end

-- 帧格式见 docs/protocol.md：第一行 7 个字段，第二行 OBS_DIM 个数值
local SCHEMA = "obs-v0.1"
local TICK = 0
local EPISODE = 0
local REWARD, TERM, TRUNC = 0, 0, 0

-- 任务脚本通过这些函数写奖励与终止；不要在 Python 侧重算
function ced_obs_set_reward(r) REWARD = r end
function ced_obs_set_termination() TERM = 1 end
function ced_obs_set_truncation() TRUNC = 1 end
function ced_obs_new_episode() EPISODE = EPISODE + 1; REWARD, TERM, TRUNC = 0, 0, 0 end

local function write_obs(path, values)
    local head = string.format("%s %d %d %d %.6f %d %d",
        SCHEMA, TICK, EPISODE, OBS_DIM, REWARD, TERM, TRUNC)
    local tmp = path .. ".tmp"
    local f = io.open(tmp, "w")
    if not f then return end
    f:write(head .. "\n" .. table.concat(values, " ") .. "\n")
    f:close()
    os.rename(tmp, path)                       -- 原子替换，Python 不会读到半写文件
end

-- TODO(day-6): 多环境时按假人名后缀区分；Luanti 的 mod 沙箱里没有 os.getenv
local TARGET = "ced_bot"
local initialized = false

core.register_globalstep(function(_)
    local player = core.get_player_by_name(TARGET)
    if not player then return end
    if not initialized then
        core.after(1.0, function()
            core.chat_send_player(TARGET, "ced_obs ready, obs_dim=" .. OBS_DIM)
        end)
        initialized = true
    end
    TICK = TICK + 1
    local path = core.get_worldpath() .. "/ced_obs_" .. TARGET .. ".txt"
    write_obs(path, build_obs(player))
end)

-- TODO(day-6)：动作通道。Lua 每 tick 读 act.txt 的 `<tick> <14 ints>`，
-- 用 core.set_player_controls 之类的接口施加，并在应用后记录 last_applied。
-- 先不做，是因为还没有接 Craftium 的控制器；接口约定见 docs/protocol.md。
