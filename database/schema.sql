-- Smart City Traffic AI - PostgreSQL + PostGIS schema
-- Phase 15 / Step 15
--
-- Stores historical traffic telemetry, vehicle trajectories, emissions,
-- signal states, and RL reward/action traces from the SUMO/PPO stack.

BEGIN;

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS traffic;
SET search_path TO traffic, public;

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'controller_mode') THEN
        CREATE TYPE controller_mode AS ENUM ('ppo', 'webster', 'manual', 'emergency', 'bus_priority', 'offline');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_severity') THEN
        CREATE TYPE alert_severity AS ENUM ('info', 'warning', 'critical');
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Simulation sessions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS simulation_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    scenario_name TEXT NOT NULL DEFAULT 'dhanbad',
    sumo_config_path TEXT,
    network_path TEXT,
    route_path TEXT,
    controller controller_mode NOT NULL DEFAULT 'ppo',
    ai_enabled BOOLEAN NOT NULL DEFAULT true,
    model_path TEXT,
    model_sha256 TEXT,
    device TEXT,
    step_length_s NUMERIC(8, 3) NOT NULL DEFAULT 1.0,
    notes TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK (step_length_s > 0)
);

-- ---------------------------------------------------------------------------
-- Network dimensions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intersections (
    tls_id TEXT PRIMARY KEY,
    name TEXT,
    geom GEOMETRY(Point, 4326),
    sumo_x NUMERIC(12, 3),
    sumo_y NUMERIC(12, 3),
    controlled_lanes TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lanes (
    lane_id TEXT PRIMARY KEY,
    edge_id TEXT,
    tls_id TEXT REFERENCES intersections(tls_id) ON DELETE SET NULL,
    length_m NUMERIC(10, 2),
    speed_limit_mps NUMERIC(8, 3),
    shape GEOMETRY(LineString, 4326),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length_m IS NULL OR length_m >= 0),
    CHECK (speed_limit_mps IS NULL OR speed_limit_mps >= 0)
);

-- ---------------------------------------------------------------------------
-- Historical traffic telemetry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS traffic_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    sim_time_s NUMERIC(14, 3) NOT NULL,
    total_vehicles INTEGER NOT NULL DEFAULT 0,
    total_queue INTEGER NOT NULL DEFAULT 0,
    total_waiting_s NUMERIC(14, 3) NOT NULL DEFAULT 0,
    total_co2_g NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_rate_g_s NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_reduction_pct NUMERIC(7, 3),
    ai_enabled BOOLEAN NOT NULL DEFAULT true,
    device TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (session_id, sim_step),
    CHECK (sim_step >= 0),
    CHECK (sim_time_s >= 0),
    CHECK (total_vehicles >= 0),
    CHECK (total_queue >= 0)
);

CREATE TABLE IF NOT EXISTS intersection_snapshots (
    intersection_snapshot_id BIGSERIAL PRIMARY KEY,
    snapshot_id BIGINT NOT NULL REFERENCES traffic_snapshots(snapshot_id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    tls_id TEXT NOT NULL REFERENCES intersections(tls_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    sim_step INTEGER NOT NULL,
    phase_index INTEGER,
    phase_state TEXT,
    phase_duration_s NUMERIC(10, 3),
    total_queue INTEGER NOT NULL DEFAULT 0,
    total_vehicles INTEGER NOT NULL DEFAULT 0,
    total_waiting_s NUMERIC(14, 3) NOT NULL DEFAULT 0,
    total_co2_g NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_rate_g_s NUMERIC(14, 6) NOT NULL DEFAULT 0,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (snapshot_id, tls_id),
    CHECK (total_queue >= 0),
    CHECK (total_vehicles >= 0)
);

CREATE TABLE IF NOT EXISTS lane_snapshots (
    lane_snapshot_id BIGSERIAL PRIMARY KEY,
    intersection_snapshot_id BIGINT NOT NULL REFERENCES intersection_snapshots(intersection_snapshot_id) ON DELETE CASCADE,
    snapshot_id BIGINT NOT NULL REFERENCES traffic_snapshots(snapshot_id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    lane_id TEXT NOT NULL REFERENCES lanes(lane_id) ON DELETE CASCADE,
    tls_id TEXT REFERENCES intersections(tls_id) ON DELETE SET NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    sim_step INTEGER NOT NULL,
    queue_count INTEGER NOT NULL DEFAULT 0,
    vehicle_count INTEGER NOT NULL DEFAULT 0,
    mean_speed_mps NUMERIC(10, 4) NOT NULL DEFAULT 0,
    occupancy_pct NUMERIC(7, 3) NOT NULL DEFAULT 0,
    waiting_time_s NUMERIC(14, 3) NOT NULL DEFAULT 0,
    class_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    co2_g NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_rate_g_s NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_per_vehicle_g NUMERIC(14, 6) NOT NULL DEFAULT 0,
    emission_intensity NUMERIC(7, 4) NOT NULL DEFAULT 0,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (snapshot_id, lane_id),
    CHECK (queue_count >= 0),
    CHECK (vehicle_count >= 0),
    CHECK (mean_speed_mps >= 0),
    CHECK (occupancy_pct >= 0)
);

-- ---------------------------------------------------------------------------
-- Vehicle trajectories
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id TEXT PRIMARY KEY,
    vehicle_type TEXT,
    vehicle_class TEXT,
    route_id TEXT,
    first_seen_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS vehicle_trajectories (
    trajectory_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    sim_time_s NUMERIC(14, 3) NOT NULL,
    lane_id TEXT REFERENCES lanes(lane_id) ON DELETE SET NULL,
    edge_id TEXT,
    route_index INTEGER,
    speed_mps NUMERIC(10, 4) NOT NULL DEFAULT 0,
    acceleration_mps2 NUMERIC(10, 4),
    waiting_time_s NUMERIC(14, 3) NOT NULL DEFAULT 0,
    distance_m NUMERIC(14, 3),
    heading_deg NUMERIC(8, 3),
    position GEOMETRY(Point, 4326),
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (session_id, vehicle_id, sim_step),
    CHECK (sim_step >= 0),
    CHECK (sim_time_s >= 0),
    CHECK (speed_mps >= 0)
);

-- ---------------------------------------------------------------------------
-- Emissions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS emission_records (
    emission_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    snapshot_id BIGINT REFERENCES traffic_snapshots(snapshot_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('network', 'intersection', 'lane', 'vehicle')),
    tls_id TEXT REFERENCES intersections(tls_id) ON DELETE SET NULL,
    lane_id TEXT REFERENCES lanes(lane_id) ON DELETE SET NULL,
    vehicle_id TEXT REFERENCES vehicles(vehicle_id) ON DELETE SET NULL,
    co2_g NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_rate_g_s NUMERIC(14, 6) NOT NULL DEFAULT 0,
    co2_per_vehicle_g NUMERIC(14, 6),
    emission_factor_g_km NUMERIC(14, 6),
    distance_km NUMERIC(14, 6),
    emission_intensity NUMERIC(7, 4),
    model_name TEXT NOT NULL DEFAULT 'COPERT_IV',
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (sim_step >= 0),
    CHECK (co2_g >= 0),
    CHECK (co2_rate_g_s >= 0)
);

CREATE TABLE IF NOT EXISTS pollution_hotspots (
    hotspot_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    snapshot_id BIGINT REFERENCES traffic_snapshots(snapshot_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    tls_id TEXT REFERENCES intersections(tls_id) ON DELETE SET NULL,
    lane_id TEXT REFERENCES lanes(lane_id) ON DELETE SET NULL,
    geom GEOMETRY(Point, 4326),
    co2_rate_g_s NUMERIC(14, 6) NOT NULL DEFAULT 0,
    emission_intensity NUMERIC(7, 4) NOT NULL DEFAULT 0,
    rank INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (rank > 0),
    CHECK (co2_rate_g_s >= 0)
);

-- ---------------------------------------------------------------------------
-- Signal states and control actions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signal_states (
    signal_state_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    snapshot_id BIGINT REFERENCES traffic_snapshots(snapshot_id) ON DELETE CASCADE,
    tls_id TEXT NOT NULL REFERENCES intersections(tls_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    phase_index INTEGER,
    phase_state TEXT,
    phase_duration_s NUMERIC(10, 3),
    green_count INTEGER NOT NULL DEFAULT 0,
    yellow_count INTEGER NOT NULL DEFAULT 0,
    red_count INTEGER NOT NULL DEFAULT 0,
    is_green BOOLEAN NOT NULL DEFAULT false,
    controller controller_mode NOT NULL DEFAULT 'ppo',
    action INTEGER,
    action_source TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (session_id, tls_id, sim_step),
    CHECK (sim_step >= 0)
);

CREATE TABLE IF NOT EXISTS signal_actions (
    signal_action_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    tls_id TEXT NOT NULL REFERENCES intersections(tls_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    action INTEGER NOT NULL,
    controller controller_mode NOT NULL DEFAULT 'ppo',
    previous_phase_index INTEGER,
    next_phase_index INTEGER,
    reason TEXT,
    emergency_vehicle_id TEXT REFERENCES vehicles(vehicle_id) ON DELETE SET NULL,
    bus_vehicle_id TEXT REFERENCES vehicles(vehicle_id) ON DELETE SET NULL,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (action BETWEEN 0 AND 3),
    CHECK (sim_step >= 0)
);

-- ---------------------------------------------------------------------------
-- RL rewards and training/evaluation traces
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rl_rewards (
    reward_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    tls_id TEXT REFERENCES intersections(tls_id) ON DELETE SET NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_step INTEGER NOT NULL,
    episode INTEGER,
    action INTEGER,
    reward NUMERIC(14, 6) NOT NULL,
    cumulative_reward NUMERIC(14, 6),
    queue_penalty NUMERIC(14, 6),
    wait_delta_penalty NUMERIC(14, 6),
    co2_penalty NUMERIC(14, 6),
    bus_priority_bonus NUMERIC(14, 6),
    emergency_priority_bonus NUMERIC(14, 6),
    observation JSONB NOT NULL DEFAULT '{}'::jsonb,
    info JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (sim_step >= 0),
    CHECK (action IS NULL OR action BETWEEN 0 AND 3)
);

CREATE TABLE IF NOT EXISTS rl_episodes (
    episode_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    episode INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    steps INTEGER NOT NULL DEFAULT 0,
    total_reward NUMERIC(14, 6) NOT NULL DEFAULT 0,
    mean_queue NUMERIC(14, 6),
    mean_waiting_s NUMERIC(14, 6),
    mean_co2_rate_g_s NUMERIC(14, 6),
    terminated BOOLEAN,
    truncated BOOLEAN,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (session_id, episode),
    CHECK (steps >= 0),
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

-- ---------------------------------------------------------------------------
-- Alerts and analytics events
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS traffic_alerts (
    alert_id TEXT PRIMARY KEY,
    session_id UUID REFERENCES simulation_sessions(session_id) ON DELETE CASCADE,
    tls_id TEXT REFERENCES intersections(tls_id) ON DELETE SET NULL,
    severity alert_severity NOT NULL,
    message TEXT NOT NULL,
    queue_count INTEGER,
    waiting_time_s NUMERIC(14, 3),
    opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    cleared_at TIMESTAMPTZ,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (cleared_at IS NULL OR cleared_at >= opened_at)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_intersections_geom
    ON intersections USING gist (geom);

CREATE INDEX IF NOT EXISTS idx_lanes_tls
    ON lanes (tls_id);

CREATE INDEX IF NOT EXISTS idx_lanes_shape
    ON lanes USING gist (shape);

CREATE INDEX IF NOT EXISTS idx_traffic_snapshots_session_step
    ON traffic_snapshots (session_id, sim_step DESC);

CREATE INDEX IF NOT EXISTS idx_traffic_snapshots_recorded_brin
    ON traffic_snapshots USING brin (recorded_at);

CREATE INDEX IF NOT EXISTS idx_intersection_snapshots_tls_time
    ON intersection_snapshots (tls_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_lane_snapshots_lane_time
    ON lane_snapshots (lane_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_trajectories_vehicle_step
    ON vehicle_trajectories (vehicle_id, sim_step DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_trajectories_session_step
    ON vehicle_trajectories (session_id, sim_step DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_trajectories_position
    ON vehicle_trajectories USING gist (position);

CREATE INDEX IF NOT EXISTS idx_emission_records_scope_time
    ON emission_records (scope, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_emission_records_tls_time
    ON emission_records (tls_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_pollution_hotspots_geom
    ON pollution_hotspots USING gist (geom);

CREATE INDEX IF NOT EXISTS idx_pollution_hotspots_rank
    ON pollution_hotspots (session_id, sim_step DESC, rank);

CREATE INDEX IF NOT EXISTS idx_signal_states_tls_step
    ON signal_states (tls_id, sim_step DESC);

CREATE INDEX IF NOT EXISTS idx_signal_actions_tls_step
    ON signal_actions (tls_id, sim_step DESC);

CREATE INDEX IF NOT EXISTS idx_rl_rewards_session_step
    ON rl_rewards (session_id, sim_step DESC);

CREATE INDEX IF NOT EXISTS idx_rl_episodes_session_episode
    ON rl_episodes (session_id, episode DESC);

CREATE INDEX IF NOT EXISTS idx_traffic_alerts_open
    ON traffic_alerts (severity, opened_at DESC)
    WHERE cleared_at IS NULL;

-- ---------------------------------------------------------------------------
-- Convenience views
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW latest_network_snapshot AS
SELECT DISTINCT ON (session_id)
    session_id,
    snapshot_id,
    recorded_at,
    sim_step,
    total_vehicles,
    total_queue,
    total_waiting_s,
    total_co2_g,
    co2_rate_g_s,
    co2_reduction_pct
FROM traffic_snapshots
ORDER BY session_id, sim_step DESC;

CREATE OR REPLACE VIEW latest_pollution_hotspots AS
SELECT DISTINCT ON (session_id, tls_id)
    session_id,
    tls_id,
    lane_id,
    recorded_at,
    sim_step,
    co2_rate_g_s,
    emission_intensity,
    rank,
    geom
FROM pollution_hotspots
ORDER BY session_id, tls_id, sim_step DESC, rank ASC;

CREATE OR REPLACE VIEW intersection_hourly_emissions AS
SELECT
    session_id,
    tls_id,
    date_trunc('hour', recorded_at) AS hour_bucket,
    avg(co2_rate_g_s) AS avg_co2_rate_g_s,
    max(co2_rate_g_s) AS peak_co2_rate_g_s,
    avg(total_queue) AS avg_queue,
    avg(total_waiting_s) AS avg_waiting_s
FROM intersection_snapshots
GROUP BY session_id, tls_id, date_trunc('hour', recorded_at);

COMMIT;
