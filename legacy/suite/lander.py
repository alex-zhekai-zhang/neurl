"""One configurable LunarLander (continuous actions), derived from Gymnasium's lunar_lander.py.

    layout   "stock"  main + side engines, action [main, lateral]; identical to LunarLander-v3 continuous
             "angled" two 45-degree boosters + side engines, action [lateral, left booster, right booster]
    pads     "center" one pad, stock reward
             "two"    left/right pads; rest on env.good_pad +100, on the other -100, off both 0
    pad_shaping   "nearer": shaping reward pulls toward the nearer pad; "good": toward env.good_pad
    off_pad_reward   reward for coming to rest off both pads (two pads only)
    engine_scale   {engine: power multiplier}; also settable between episodes via env.engine_scale[name]
    fault          (step, engine, scale): the engine's scale changes at that step of every episode
    action_perm / action_sign   agent action a is executed as (a * sign)[perm]
    observe_context   append engine scales (and the good pad, -1 left / +1 right) to the observation
    rest_steps        also count as come to rest when both legs touch the ground this many steps in a row

info["outcome"] is set on termination: crashed, out_of_bounds, landed, landed_good, landed_bad, landed_off.
"""
__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

# Two angled boosters replace the main engine. Exhaust points ANGLED_ENGINE_ANGLE below horizontal and outward.
ANGLED_ENGINE_ANGLE = math.radians(45)
ANGLED_ENGINE_X = 10   # mounted at (-X, Y) and (+X, Y) on the lander body
ANGLED_ENGINE_Y = -10  # bottom edge of LANDER_POLY
# impulse per booster at full power: both together give the same lift as the stock main engine (13 * 4 / 30)
ANGLED_ENGINE_POWER = MAIN_ENGINE_POWER * MAIN_ENGINE_Y_LOCATION / SCALE / math.sqrt(2)
# fuel cost per booster at full power, same cost per unit impulse as the stock main engine (0.30)
ANGLED_ENGINE_FUEL = 0.30 / math.sqrt(2)

VIEWPORT_W = 600
VIEWPORT_H = 400

LAYOUTS = {
    "stock": ("main", "left_side", "right_side"),
    "angled": ("left_booster", "right_booster", "left_side", "right_side"),
}
PAD_CHUNKS = (2, 8)  # terrain chunks the left and right pads are centred on (of 11)
PAD_REWARD = 100


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        layout: str = "angled",
        pads: str = "center",
        good_pad: str = "left",
        pad_shaping: str = "nearer",
        off_pad_reward: float = 0.0,
        engine_scale: dict | None = None,
        fault: tuple | None = None,
        action_perm: tuple | None = None,
        action_sign: tuple | None = None,
        observe_context: bool = False,
        rest_steps: int | None = None,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
            layout,
            pads,
            good_pad,
            pad_shaping,
            off_pad_reward,
            engine_scale,
            fault,
            action_perm,
            action_sign,
            observe_context,
            rest_steps,
        )

        assert -12.0 < gravity and gravity < 0.0, (
            f"gravity (current value: {gravity}) must be between -12 and 0"
        )
        self.gravity = gravity

        assert layout in LAYOUTS and pads in ("center", "two") and good_pad in ("left", "right")
        self.layout = layout
        self.two_pads = pads == "two"
        self.good_pad = good_pad
        assert pad_shaping in ("nearer", "good")
        self.pad_shaping = pad_shaping
        self.off_pad_reward = off_pad_reward
        self.engines = LAYOUTS[layout]
        self.engine_scale = {name: 1.0 for name in self.engines}
        self.engine_scale.update(engine_scale or {})
        assert set(self.engine_scale) == set(self.engines), f"engines are {self.engines}"
        self.fault = fault
        self._fault_scale = {}
        self._t = 0
        act_dim = 2 if layout == "stock" else 3
        self.action_perm = list(action_perm) if action_perm is not None else list(range(act_dim))
        self.action_sign = np.asarray(action_sign if action_sign is not None else [1.0] * act_dim, dtype=np.float64)
        assert sorted(self.action_perm) == list(range(act_dim)) and len(self.action_sign) == act_dim
        self.observe_context = observe_context
        self.rest_steps = rest_steps
        self._rest = 0

        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Box2D.b2Body | None = None
        self.particles = []

        self.prev_reward = None

        self.continuous = True
        n_context = (len(self.engines) + int(self.two_pads)) if observe_context else 0

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        low = np.concatenate([low, np.full(n_context, -1.0, dtype=np.float32)])
        high = np.concatenate([high, np.full(n_context, 2.0, dtype=np.float32)])
        self.observation_space = spaces.Box(low, high)

        # stock:  [main, lateral]; angled: [lateral, left booster, right booster]
        # main / booster: -1..0 off, 0..+1 throttle from 50% to 100% power
        # lateral: -1.0..-0.5 fire left side engine, +0.5..+1.0 fire right side engine, -0.5..0.5 off
        self.action_space = spaces.Box(-1, +1, (act_dim,), dtype=np.float32)

        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()

        # Bug's workaround for: https://github.com/Farama-Foundation/Gymnasium/issues/728
        # Not sure why the self._destroy() is not enough to clean(reset) the total world environment elements, need more investigation on the root cause,
        # we must create a totally new world for self.reset(), or the bug#728 will happen
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_y = H / 4
        # name -> (x1, x2); each pad spans one chunk either side of its centre chunk
        centres = dict(zip(("left", "right"), PAD_CHUNKS)) if self.two_pads else {"center": CHUNKS // 2}
        self.pads = {}
        for name, c in centres.items():
            self.pads[name] = (chunk_x[c - 1], chunk_x[c + 1])
            for i in range(c - 2, c + 3):
                height[i] = self.helipad_y
        # pad centres in the units of state[0]
        self.pad_centres = {n: ((x1 + x2) / 2 - W / 2) / (W / 2) if self.two_pads else 0.0 for n, (x1, x2) in self.pads.items()}
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:  # Initialize wind pattern based on index
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()
        self._fault_scale = {}
        obs = self.step(np.zeros(self.action_space.shape[0]))[0]
        self._t = 0
        self._rest = 0
        self._fault_scale = {}
        return obs, {}

    def _scale(self, engine):
        return self._fault_scale.get(engine, self.engine_scale[engine])

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        action = np.clip(action, -1, +1).astype(np.float64)
        action = (action * self.action_sign)[self.action_perm]
        if self.fault is not None and self._t == self.fault[0]:
            self._fault_scale[self.fault[1]] = self.fault[2]
        self._t += 1
        lateral = action[1] if self.layout == "stock" else action[0]

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # Angled boosters: action[1] = left booster, action[2] = right booster.
        # Exhaust points down and outward, so the thrust pushes the lander up and toward the opposite side.
        # engines fired this step, for episode_recorder.py: (x, y, exhaust_dx, exhaust_dy, power, kind),
        # kind 0 = big booster, 1 = side engine. No effect on the physics.
        self.flames = []

        m_power = 0.0
        if self.layout == "stock" and action[0] > 0.0:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5 * self._scale("main")  # 0.5..1.0
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse(
                    (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.flames.append((impulse_pos[0], impulse_pos[1], ox, oy, m_power, 0))

        a_power = [0.0, 0.0]
        for k, sign in enumerate((-1, +1)):
            if self.layout == "stock" or action[1 + k] <= 0.0:
                continue
            a_power[k] = (np.clip(action[1 + k], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            a_power[k] *= self._scale(("left_booster", "right_booster")[k])

            # exhaust direction in world coordinates, jittered by dispersion like the stock engines
            exhaust = self.lander.GetWorldVector(
                (
                    sign * math.cos(ANGLED_ENGINE_ANGLE) + dispersion[0],
                    -math.sin(ANGLED_ENGINE_ANGLE) + dispersion[1],
                )
            )
            impulse_pos = self.lander.GetWorldPoint(
                (sign * ANGLED_ENGINE_X / SCALE, ANGLED_ENGINE_Y / SCALE)
            )
            impulse = ANGLED_ENGINE_POWER * a_power[k]
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], a_power[k])
                p.ApplyLinearImpulse(
                    (exhaust[0] * impulse, exhaust[1] * impulse), impulse_pos, True
                )
            self.lander.ApplyLinearImpulse(
                (-exhaust[0] * impulse, -exhaust[1] * impulse), impulse_pos, True
            )
            self.flames.append((impulse_pos[0], impulse_pos[1], exhaust[0], exhaust[1], a_power[k], 0))

        s_power = 0.0
        if np.abs(lateral) > 0.5:
            # Orientation/Side engines: lateral < -0.5 is left, > 0.5 is right
            direction = np.sign(lateral)
            s_power = np.clip(np.abs(lateral), 0.5, 1.0)
            s_power *= self._scale("left_side" if direction < 0 else "right_side")

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.flames.append((impulse_pos[0], impulse_pos[1], ox, oy, s_power, 1))

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        # distance to the nearer pad, or to the good pad
        targets = [self.pad_centres[self.good_pad]] if self.two_pads and self.pad_shaping == "good" else self.pad_centres.values()
        pad_dist = min(np.sqrt((state[0] - cx) * (state[0] - cx) + state[1] * state[1]) for cx in targets)

        reward = 0
        shaping = (
            -100 * pad_dist
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # less fuel spent is better
        reward -= m_power * 0.30 if self.layout == "stock" else sum(a_power) * ANGLED_ENGINE_FUEL
        reward -= s_power * 0.03

        terminated = False
        info = {}
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
            info["outcome"] = "crashed" if self.game_over else "out_of_bounds"
        self._rest = self._rest + 1 if state[6] and state[7] else 0
        if not self.lander.awake or (self.rest_steps is not None and self._rest >= self.rest_steps and not terminated):
            terminated = True
            reward = +100
            info["outcome"] = "landed"
            if self.two_pads:
                x = self.lander.position.x
                pad = next((n for n, (x1, x2) in self.pads.items() if x1 <= x <= x2), None)
                reward = self.off_pad_reward if pad is None else (+PAD_REWARD if pad == self.good_pad else -PAD_REWARD)
                info["outcome"] = "landed_off" if pad is None else ("landed_good" if pad == self.good_pad else "landed_bad")
                info["pad"] = pad

        if self.observe_context:
            state += [self._scale(e) for e in self.engines]
            if self.two_pads:
                state.append(-1.0 if self.good_pad == "left" else 1.0)

        if self.render_mode == "human":
            self.render()
        # truncation=False: the caller applies the time limit
        return np.array(state, dtype=np.float32), reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x, flag_color in [
                    (x, (204, 204, 0) if not self.two_pads else (0, 200, 0) if name == self.good_pad else (220, 50, 50))
                    for name, xs in self.pads.items()
                    for x in xs
                ]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=flag_color,
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        flag_color,
                    )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
