#!/usr/bin/env python3
"""
JAX Ball Simulation Engine
==========================
Point mass under gravity, linear air drag, and adjustable engine thrust.

Physics:  JAX JIT-compiled, semi-implicit (symplectic) Euler integration
Display:  Matplotlib real-time animation
HUD:      Top-right inset compass + telemetry text (velocity in polar form)

Controls:
    ↑/↓     Thrust +/-
    ←/→     Rotate engine angle
    W/A/S/D Quick set engine direction (up/left/down/right)
    SPACE   Pause / Resume
    R       Reset
    T       Toggle trail
    F       Toggle force arrows

Usage:
    python ball_sim_jax.py
"""

import jax
jax.config.update("jax_enable_x64", True)  # float64 for physics precision

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Wedge
from dataclasses import dataclass
from typing import NamedTuple


# ================================================================
# 1. Physics Core (Pure JAX)
# ================================================================

class SimConfig(NamedTuple):
    """Physical parameters — baked into JIT as static constants."""
    g: float = 9.81        # gravitational acceleration (m/s²)
    mass: float = 1.0      # ball mass (kg)
    drag_k: float = 0.05   # linear drag coefficient (N·s/m)
    dt: float = 0.01       # time step (s)
    bounce: float = 0.5    # ground collision energy retention


@jax.jit
def physics_step(state, control, cfg):
    """
    Advance simulation by one dt using semi-implicit (symplectic) Euler.

    Args:
        state:   jnp.array [x, y, vx, vy]
        control: jnp.array [thrust (N), angle (rad)]  — traced array
        cfg:     SimConfig — static constants

    Returns:
        jnp.array [x_new, y_new, vx_new, vy_new]

    Forces:
        Gravity : F = m * g  (downward)
        Drag    : F = -k * v (linear, opposes velocity)
        Engine  : F = thrust * [cos(θ), sin(θ)]
    """
    x, y, vx, vy = state
    thrust, angle = control[0], control[1]

    # --- Net force ---
    Fx = -cfg.drag_k * vx + thrust * jnp.cos(angle)
    Fy = -cfg.drag_k * vy - cfg.mass * cfg.g + thrust * jnp.sin(angle)

    # --- Acceleration (a = F / m) ---
    ax = Fx / cfg.mass
    ay = Fy / cfg.mass

    # --- Semi-implicit Euler: update velocity first, then position ---
    vx_new = vx + ax * cfg.dt
    vy_new = vy + ay * cfg.dt
    x_new  = x  + vx_new * cfg.dt
    y_new  = y  + vy_new * cfg.dt

    # --- Ground collision (y >= 0, bounce with damping) ---
    hit = y_new <= 0.0
    y_new  = jnp.where(hit, 0.0, y_new)
    vy_new = jnp.where(hit, -vy_new * cfg.bounce, vy_new)

    return jnp.array([x_new, y_new, vx_new, vy_new])


@jax.jit
def batch_simulate(state0, thrusts, angles, cfg):
    """
    Batch trajectory computation via jax.lax.scan (no GUI).

    Args:
        state0:   jnp.array [x0, y0, vx0, vy0]
        thrusts:  jnp.array (N,) — thrust at each step
        angles:   jnp.array (N,) — engine angle at each step
        cfg:      SimConfig

    Returns:
        trajectory: jnp.array (N, 4)
    """
    def scan_fn(carry, inputs):
        t, a = inputs
        ctrl = jnp.array([t, a])
        s = physics_step(carry, ctrl, cfg)
        return s, s

    _, traj = jax.lax.scan(scan_fn, state0, (thrusts, angles))
    return traj


# ================================================================
# 2. Controller (Keyboard Input)
# ================================================================

@dataclass
class EngineState:
    thrust: float = 8.0           # Newtons
    angle: float = np.pi / 2      # radians (default: straight up)


@dataclass
class Flags:
    paused: bool = False
    reset: bool = False
    trail: bool = True
    arrows: bool = True


class KeyHandler:
    """Maps key presses to engine / flag mutations."""

    def __init__(self):
        self.engine = EngineState()
        self.flags = Flags()

    def __call__(self, event):
        k = event.key
        e, f = self.engine, self.flags

        if k == 'up':         e.thrust = min(e.thrust + 1.0, 50.0)
        elif k == 'down':     e.thrust = max(e.thrust - 1.0, 0.0)
        elif k == 'left':     e.angle += 0.1
        elif k == 'right':    e.angle -= 0.1
        elif k == 'w':        e.angle = np.pi / 2       # up
        elif k == 'a':        e.angle = np.pi           # left
        elif k == 's':        e.angle = -np.pi / 2      # down
        elif k == 'd':        e.angle = 0.0             # right
        elif k == ' ':        f.paused = not f.paused
        elif k == 'r':        f.reset = True
        elif k == 't':        f.trail = not f.trail
        elif k == 'f':        f.arrows = not f.arrows


# ================================================================
# 3. Simulator (Visualization + Main Loop)
# ================================================================

class BallSimulator:
    """Real-time JAX ball simulation with matplotlib visualization."""

    def __init__(self):
        self.cfg = SimConfig()
        self.state = jnp.array([0.0, 40.0, 8.0, 0.0])  # [x, y, vx, vy]
        self.handler = KeyHandler()
        self.trail_x, self.trail_y = [], []
        self._setup_figure()

    # ---------- Figure Setup ----------

    def _setup_figure(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('JAX Ball Simulation Engine')

        # Ground
        self.ax.axhline(0, color='sienna', lw=2, alpha=0.6)
        self.ax.fill_between([-500, 500], -10, 0, color='sienna', alpha=0.1)

        # Trail
        self.trail_line, = self.ax.plot([], [], color='cyan', alpha=0.2, lw=1)

        # Ball
        self.ball, = self.ax.plot([], [], 'o', color='#FF4444', ms=12, zorder=5)

        # Engine force arrow (orange)
        self.eng_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->',
            color='orange', mutation_scale=15, lw=2)
        self.ax.add_patch(self.eng_arrow)

        # Velocity arrow (green)
        self.vel_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->',
            color='#44FF44', mutation_scale=15, lw=2)
        self.ax.add_patch(self.vel_arrow)

        # --- Top-right inset: velocity compass "window" ---
        self.inset = self.fig.add_axes([0.74, 0.58, 0.24, 0.33])
        self.inset.set_facecolor('#0a0a1a')
        self.inset.set_xlim(-1.3, 1.3)
        self.inset.set_ylim(-1.3, 1.3)
        self.inset.set_aspect('equal')
        self.inset.set_title('Velocity (polar)',
                             color='#00FF88', fontsize=9, pad=4)
        self.inset.tick_params(colors='#444', labelsize=6)
        for spine in self.inset.spines.values():
            spine.set_color('#555')

        # Compass ring
        theta_c = np.linspace(0, 2 * np.pi, 100)
        self.inset.plot(np.cos(theta_c), np.sin(theta_c),
                        color='#333', lw=0.5)
        # Crosshairs
        self.inset.plot([-1.15, 1.15], [0, 0], color='#222', lw=0.5)
        self.inset.plot([0, 0], [-1.15, 1.15], color='#222', lw=0.5)
        # Angle labels
        for ang_deg, label in [(0, '0°'), (90, '90°'),
                                (180, '180°'), (-90, '-90°')]:
            ang_rad = np.radians(ang_deg)
            self.inset.text(
                1.12 * np.cos(ang_rad), 1.12 * np.sin(ang_rad),
                label, color='#666', fontsize=6, ha='center', va='center')

        # Velocity arrow inside compass
        self.compass_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->',
            color='#00FF88', mutation_scale=12, lw=2.5)
        self.inset.add_patch(self.compass_arrow)

        # Angle arc (filled wedge from 0° to current angle)
        self.angle_wedge = Wedge(
            (0, 0), 0.35, 0, 0,
            facecolor='#00FF88', alpha=0.15, edgecolor='none')
        self.inset.add_patch(self.angle_wedge)

        # Compass readout text
        self.compass_text = self.inset.text(
            -1.2, -1.2, '', color='#00FF88', fontsize=7,
            family='monospace', va='bottom')

        # --- HUD telemetry text (below compass) ---
        self.hud = self.ax.text(
            0.98, 0.52, '', transform=self.ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#1a1a2e', edgecolor='#444', alpha=0.9),
            color='#00FF88', family='monospace')

        # --- Controls hint (bottom-left) ---
        self.ax.text(
            0.02, 0.02,
            '↑↓ Thrust  ←→ Angle  WASD Dir  '
            'SPACE Pause  R Reset  T Trail  F Arrows',
            transform=self.ax.transAxes,
            fontsize=8, color='#666', family='monospace')

        # Axes config
        self.ax.set_xlim(-80, 80)
        self.ax.set_ylim(-5, 100)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)', color='#aaa')
        self.ax.set_ylabel('Y (m)', color='#aaa')
        self.ax.set_title('JAX Ball Simulation — Gravity + Drag + Engine',
                          color='white', fontsize=13)
        self.ax.grid(True, alpha=0.12)

        self.fig.canvas.mpl_connect('key_press_event', self.handler)

    # ---------- HUD Updates ----------

    def _update_hud(self, x, y, vx, vy):
        """Build telemetry text with polar velocity representation."""
        speed = float(jnp.sqrt(vx**2 + vy**2))
        theta = float(jnp.degrees(jnp.arctan2(vy, vx)))

        t = self.handler.engine.thrust
        ta = float(jnp.degrees(self.handler.engine.angle))
        status = "■ PAUSED " if self.handler.flags.paused else "▶ RUNNING"

        self.hud.set_text(
            f"╔════════ TELEMETRY ════════╗\n"
            f"║  {status:<20s}    ║\n"
            f"║──────────────────────────║\n"
            f"║  POSITION                ║\n"
            f"║    X  : {x:+8.2f} m      ║\n"
            f"║    Y  : {y:+8.2f} m      ║\n"
            f"║──────────────────────────║\n"
            f"║  VELOCITY (polar)        ║\n"
            f"║    |v| : {speed:7.2f} m/s ║\n"
            f"║    θ  : {theta:+7.1f}°   ║\n"
            f"║    vx : {vx:+8.2f}       ║\n"
            f"║    vy : {vy:+8.2f}       ║\n"
            f"║──────────────────────────║\n"
            f"║  ENGINE                  ║\n"
            f"║    F  : {t:7.2f} N       ║\n"
            f"║    θ  : {ta:+7.1f}°      ║\n"
            f"║──────────────────────────║\n"
            f"║  g={self.cfg.g}  m={self.cfg.mass}  k={self.cfg.drag_k}   ║\n"
            f"╚══════════════════════════╝"
        )

    def _update_compass(self, vx, vy):
        """Update inset compass: velocity vector + angle arc + readout."""
        speed = float(jnp.sqrt(vx**2 + vy**2))
        theta_deg = float(jnp.degrees(jnp.arctan2(vy, vx)))

        if speed > 0.01:
            # Scale: full compass = 30 m/s
            scale = min(speed / 30.0, 1.0)
            dx = float(vx) / speed * scale
            dy = float(vy) / speed * scale
        else:
            dx, dy = 0.0, 0.0

        self.compass_arrow.set_positions((0, 0), (dx, dy))

        # Update angle wedge (from 0° to current angle)
        self.angle_wedge.set_theta1(0)
        self.angle_wedge.set_theta2(theta_deg if theta_deg >= 0 else theta_deg + 360)

        self.compass_text.set_text(
            f"|v| = {speed:6.2f} m/s\n"
            f"θ   = {theta_deg:+6.1f}°")

    # ---------- Animation Callback ----------

    def update(self, frame):
        """One physics step + redraw."""
        # Handle reset
        if self.handler.flags.reset:
            self.state = jnp.array([0.0, 40.0, 8.0, 0.0])
            self.trail_x.clear()
            self.trail_y.clear()
            self.handler.flags.reset = False

        # Physics step (JAX JIT)
        if not self.handler.flags.paused:
            ctrl = jnp.array([self.handler.engine.thrust,
                              self.handler.engine.angle])
            self.state = physics_step(self.state, ctrl, self.cfg)

        # Extract state to Python floats for matplotlib
        x, y, vx, vy = [float(v) for v in self.state]

        # Update trail
        if self.handler.flags.trail and not self.handler.flags.paused:
            self.trail_x.append(x)
            self.trail_y.append(y)
            if len(self.trail_x) > 800:
                self.trail_x.pop(0)
                self.trail_y.pop(0)

        self.trail_line.set_data(self.trail_x, self.trail_y)
        self.ball.set_data([x], [y])

        # Force arrows
        if self.handler.flags.arrows:
            tm = self.handler.engine.thrust
            ta = self.handler.engine.angle
            arrow_len = tm * 0.8  # visual scale factor
            self.eng_arrow.set_positions(
                (x, y),
                (x + arrow_len * np.cos(ta),
                 y + arrow_len * np.sin(ta)))
            self.eng_arrow.set_visible(True)

            self.vel_arrow.set_positions(
                (x, y),
                (x + vx * 1.5, y + vy * 1.5))
            self.vel_arrow.set_visible(True)
        else:
            self.eng_arrow.set_visible(False)
            self.vel_arrow.set_visible(False)

        # HUD + compass
        self._update_hud(x, y, vx, vy)
        self._update_compass(vx, vy)

        # Camera follow (smooth scroll)
        margin = 25
        xl, xr = self.ax.get_xlim()
        if x > xr - margin:
            self.ax.set_xlim(x - 80, x + 80)
        elif x < xl + margin:
            self.ax.set_xlim(x - 80, x + 80)
        yl, yh = self.ax.get_ylim()
        if y > yh - margin:
            self.ax.set_ylim(y - 50, y + 50)

        return []

    # ---------- Run ----------

    def run(self):
        self._print_banner()
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=15,
            blit=False, cache_frame_data=False)
        plt.show()

    @staticmethod
    def _print_banner():
        print("╔══════════════════════════════════════╗")
        print("║   JAX Ball Simulation Engine v1.0   ║")
        print("╠══════════════════════════════════════╣")
        print("║ Physics:                             ║")
        print("║   • Gravity (constant g)             ║")
        print("║   • Linear air drag (F = -kv)        ║")
        print("║   • Adjustable engine thrust         ║")
        print("║   • Semi-implicit Euler (symplectic) ║")
        print("║   • JIT-compiled physics core        ║")
        print("╠══════════════════════════════════════╣")
        print("║ Controls:                            ║")
        print("║   ↑/↓    Thrust +/-                  ║")
        print("║   ←/→    Rotate engine               ║")
        print("║   W/A/S/D Quick direction            ║")
        print("║   SPACE  Pause                       ║")
        print("║   R      Reset                       ║")
        print("║   T      Toggle trail                ║")
        print("║   F      Toggle force arrows         ║")
        print("╚══════════════════════════════════════╝")
        print()


# ================================================================
# 4. Batch Demo (Headless — no GUI)
#    Demonstrates jax.lax.scan for trajectory computation
# ================================================================

def demo_batch():
    """Pure JAX batch trajectory — no visualization."""
    cfg = SimConfig()
    state0 = jnp.array([0.0, 40.0, 8.0, 0.0])

    N = 1000  # 10 seconds at dt=0.01
    t = jnp.arange(N) * cfg.dt

    # Thrust profile: constant 12N, angle oscillating
    thrusts = 12.0 * jnp.ones(N)
    angles  = jnp.pi / 2 + 0.3 * jnp.sin(2 * jnp.pi * t / 5.0)

    traj = batch_simulate(state0, thrusts, angles, cfg)

    print(f"Trajectory shape : {traj.shape}")
    print(f"Max height       : {float(jnp.max(traj[:, 1])):.2f} m")
    print(f"Max speed        : "
          f"{float(jnp.max(jnp.sqrt(traj[:, 2]**2 + traj[:, 3]**2))):.2f} m/s")
    print(f"Final position   : ({float(traj[-1, 0]):.2f}, "
          f"{float(traj[-1, 1]):.2f}) m")
    return traj


# ================================================================
# Entry Point
# ================================================================

if __name__ == '__main__':
    # Uncomment to run headless batch demo instead of GUI:
    # demo_batch()

    BallSimulator().run()
