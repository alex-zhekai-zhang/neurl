"""Two 4-engine landers joined by a rope, landing side by side on one wide pad.

Engines and physics per lander are those of lunar_booster.py (two 45-degree boosters + side engines).
    actions   (2, 3): one [lateral, left booster, right booster] row per lander
    obs       (2, 15): per lander, the stock 8 numbers with x measured from its own target, then the partner's
              position (2), velocity (2) and angle (1) relative to it, then the tether force on it (2, in lander weights)
    reward    (2,): per-lander stock shaping toward its own target and fuel cost; at the end both get -100 if
              either crashes or leaves the screen, +100 when both have come to rest
info["outcome"]: crashed, out_of_bounds or landed; info["on_target"]: per lander, at rest within the pad half
assigned to it. The caller applies the time limit.
"""
import math

import Box2D
import gymnasium as gym
import numpy as np
from Box2D.b2 import contactListener, edgeShape, fixtureDef, polygonShape, revoluteJointDef, ropeJointDef
from gymnasium import spaces

from lunar_booster import (ANGLED_ENGINE_ANGLE, ANGLED_ENGINE_FUEL, ANGLED_ENGINE_POWER, ANGLED_ENGINE_X,
                           ANGLED_ENGINE_Y, FPS, INITIAL_RANDOM, LANDER_POLY, LEG_AWAY, LEG_DOWN, LEG_H,
                           LEG_SPRING_TORQUE, LEG_W, SCALE, SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT,
                           SIDE_ENGINE_POWER, VIEWPORT_H, VIEWPORT_W)

N = 2
TETHER_LENGTH = 3.0    # rope length between the two body centres (world units; lander body is 34/30 wide)
START_GAP = 2.4        # initial distance between the landers (rope starts slack)
TARGET_OFFSET = 1.5    # each lander's target is this far left/right of the pad centre
PAD_CHUNKS = (3, 7)    # pad spans these terrain chunk edges (of 11): 8 units wide, centred
LANDERS_COLLIDE = True  # landers can hit each other; a hit on a lander's body is a crash
COLORS = [((128, 102, 230), (77, 77, 128)), ((230, 140, 60), (128, 77, 40))]


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def _set(self, contact, touching):
        a, b = contact.fixtureA.body, contact.fixtureB.body
        for i in range(N):
            if touching and self.env.landers[i] in (a, b):
                self.env.crashed[i] = True
            for leg in self.env.legs[i]:
                if leg in (a, b) and self.env.moon in (a, b):  # only the ground counts as leg contact
                    leg.ground_contact = touching

    def BeginContact(self, contact):
        self._set(contact, True)

    def EndContact(self, contact):
        self._set(contact, False)


class TetheredLanders(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5):
        assert -12.0 < gravity < 0.0
        self.gravity, self.render_mode = gravity, render_mode
        self.enable_wind, self.wind_power, self.turbulence_power = enable_wind, wind_power, turbulence_power
        self.world, self.moon, self.landers, self.legs, self.tether = None, None, [], [], None
        self.action_space = spaces.Box(-1, +1, (N, 3), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (N, 15), dtype=np.float32)

    def _destroy(self):
        if self.world is not None:
            self.world.contactListener = None
        self.world, self.moon, self.landers, self.legs, self.tether = None, None, [], [], None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.crashed = [False] * N
        self.prev_shaping = [None] * N
        W, H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE

        # terrain: random heights, flat around the pad
        chunks = 11
        height = self.np_random.uniform(0, H / 2, size=(chunks + 1,))
        chunk_x = [W / (chunks - 1) * i for i in range(chunks)]
        self.helipad_y = H / 4
        self.pad = (chunk_x[PAD_CHUNKS[0]], chunk_x[PAD_CHUNKS[1]])
        for i in range(PAD_CHUNKS[0] - 1, PAD_CHUNKS[1] + 2):
            height[i] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i] + height[i + 1]) for i in range(chunks)]
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(chunks - 1):
            p1, p2 = (chunk_x[i], smooth_y[i]), (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.targets = [W / 2 - TARGET_OFFSET, W / 2 + TARGET_OFFSET]

        # landers: same bodies and legs as lunar_booster.py, each with its own random push
        mask = 0x0011 if LANDERS_COLLIDE else 0x0001
        for k in range(N):
            x0, y0 = W / 2 + (k - (N - 1) / 2) * START_GAP, H
            group = -(k + 1)  # a lander never collides with its own legs
            lander = self.world.CreateDynamicBody(
                position=(x0, y0), angle=0.0,
                fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                                    density=5.0, friction=0.1, categoryBits=0x0010, maskBits=mask,
                                    groupIndex=group, restitution=0.0))
            lander.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                                       self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), True)
            legs = []
            for i in (-1, +1):
                leg = self.world.CreateDynamicBody(
                    position=(x0 - i * LEG_AWAY / SCALE, y0), angle=i * 0.05,
                    fixtures=fixtureDef(shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)), density=1.0,
                                        restitution=0.0, categoryBits=0x0010, maskBits=mask, groupIndex=group))
                leg.ground_contact = False
                rjd = revoluteJointDef(bodyA=lander, bodyB=leg, localAnchorA=(0, 0),
                                       localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE), enableMotor=True,
                                       enableLimit=True, maxMotorTorque=LEG_SPRING_TORQUE, motorSpeed=+0.3 * i)
                rjd.lowerAngle, rjd.upperAngle = (+0.9 - 0.5, +0.9) if i == -1 else (-0.9, -0.9 + 0.5)
                leg.joint = self.world.CreateJoint(rjd)
                legs.append(leg)
            self.landers.append(lander)
            self.legs.append(legs)

        self.tether = self.world.CreateJoint(ropeJointDef(
            bodyA=self.landers[0], bodyB=self.landers[1], localAnchorA=(0, 0), localAnchorB=(0, 0),
            maxLength=TETHER_LENGTH))
        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)
        return self.step(np.zeros((N, 3)))[0], {}

    def _fire(self, lander, action, flames):
        """Apply one lander's engines as in lunar_booster.py; returns its fuel cost.

        Appends each firing to flames as (x, y, exhaust_dx, exhaust_dy, power, kind) for episode_recorder.py,
        kind 0 = booster, 1 = side engine. No effect on the physics.
        """
        tip = (math.sin(lander.angle), math.cos(lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        a_power = [0.0, 0.0]
        for k, sign in enumerate((-1, +1)):
            if action[1 + k] <= 0.0:
                continue
            a_power[k] = (np.clip(action[1 + k], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            exhaust = lander.GetWorldVector((sign * math.cos(ANGLED_ENGINE_ANGLE) + dispersion[0],
                                             -math.sin(ANGLED_ENGINE_ANGLE) + dispersion[1]))
            pos = lander.GetWorldPoint((sign * ANGLED_ENGINE_X / SCALE, ANGLED_ENGINE_Y / SCALE))
            impulse = ANGLED_ENGINE_POWER * a_power[k]
            lander.ApplyLinearImpulse((-exhaust[0] * impulse, -exhaust[1] * impulse), pos, True)
            flames.append((pos[0], pos[1], exhaust[0], exhaust[1], a_power[k], 0))
        s_power = 0.0
        if abs(action[0]) > 0.5:
            direction = np.sign(action[0])
            s_power = np.clip(abs(action[0]), 0.5, 1.0)
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            pos = (lander.position[0] + ox - tip[0] * 17 / SCALE,
                   lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), pos, True)
            flames.append((pos[0], pos[1], ox, oy, s_power, 1))
        return sum(a_power) * ANGLED_ENGINE_FUEL + s_power * 0.03

    def _wind(self):
        if not self.enable_wind:
            return
        wind = math.tanh(math.sin(0.02 * self.wind_idx) + math.sin(math.pi * 0.01 * self.wind_idx)) * self.wind_power
        torque = math.tanh(math.sin(0.02 * self.torque_idx) + math.sin(math.pi * 0.01 * self.torque_idx)) * self.turbulence_power
        self.wind_idx += 1
        self.torque_idx += 1
        for lander, legs in zip(self.landers, self.legs):
            if not (legs[0].ground_contact or legs[1].ground_contact):
                lander.ApplyForceToCenter((wind, 0.0), True)
                lander.ApplyTorque(torque, True)

    def step(self, actions):
        actions = np.clip(np.asarray(actions, dtype=np.float64).reshape(N, 3), -1, +1)
        self._wind()
        self.flames = [[] for _ in range(N)]
        fuel = [self._fire(lander, a, fl) for lander, a, fl in zip(self.landers, actions, self.flames)]
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        W, H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
        force = self.tether.GetReactionForce(FPS)  # on landers[1]; landers[0] gets the opposite
        obs, rewards, out = [], np.zeros(N), [False] * N
        for i, (lander, legs) in enumerate(zip(self.landers, self.legs)):
            j = 1 - i
            p, v, q, u = lander.position, lander.linearVelocity, self.landers[j].position, self.landers[j].linearVelocity
            s = [
                (p.x - self.targets[i]) / (W / 2),
                (p.y - (self.helipad_y + LEG_DOWN / SCALE)) / (H / 2),
                v.x * (W / 2) / FPS,
                v.y * (H / 2) / FPS,
                lander.angle,
                20.0 * lander.angularVelocity / FPS,
                1.0 if legs[0].ground_contact else 0.0,
                1.0 if legs[1].ground_contact else 0.0,
            ]
            weight = lander.mass * abs(self.gravity)
            f = (1 if i == 1 else -1) * np.array([force.x, force.y]) / weight
            s += [(q.x - p.x) / (W / 2), (q.y - p.y) / (H / 2), (u.x - v.x) * (W / 2) / FPS,
                  (u.y - v.y) * (H / 2) / FPS, self.landers[j].angle, f[0], f[1]]
            obs.append(s)
            shaping = (-100 * np.sqrt(s[0] ** 2 + s[1] ** 2) - 100 * np.sqrt(s[2] ** 2 + s[3] ** 2)
                       - 100 * abs(s[4]) + 10 * s[6] + 10 * s[7])
            if self.prev_shaping[i] is not None:
                rewards[i] = shaping - self.prev_shaping[i]
            self.prev_shaping[i] = shaping
            rewards[i] -= fuel[i]
            out[i] = abs((p.x - W / 2) / (W / 2)) >= 1.0

        terminated, info = False, {}
        if any(self.crashed) or any(out):
            terminated, rewards[:] = True, -100
            info["outcome"] = "crashed" if any(self.crashed) else "out_of_bounds"
        elif not any(lander.awake for lander in self.landers):
            terminated, rewards[:] = True, +100
            info["outcome"] = "landed"
            half = (self.pad[0] + self.pad[1]) / 2
            bounds = [(self.pad[0], half), (half, self.pad[1])]
            info["on_target"] = [lo <= lander.position.x <= hi for lander, (lo, hi) in zip(self.landers, bounds)]
        return np.array(obs, dtype=np.float32), rewards, terminated, False, info

    def render(self):
        import pygame
        from pygame import gfxdraw

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        surf.fill((255, 255, 255))
        for poly in self.sky_polys:
            pts = [(x * SCALE, y * SCALE) for x, y in poly]
            pygame.draw.polygon(surf, (0, 0, 0), pts)
            gfxdraw.aapolygon(surf, pts, (0, 0, 0))
        a, b = (self.landers[k].position * SCALE for k in range(N))
        taut = abs((self.landers[0].position - self.landers[1].position).length - TETHER_LENGTH) < 0.02
        pygame.draw.line(surf, (200, 30, 30) if taut else (120, 120, 120), a, b, 2)
        for k in range(N):
            for body in [self.landers[k], *self.legs[k]]:
                for f in body.fixtures:
                    pts = [body.transform * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(surf, COLORS[k][0], pts)
                    gfxdraw.aapolygon(surf, pts, COLORS[k][0])
                    pygame.draw.aalines(surf, COLORS[k][1], True, pts)
            tx = self.targets[k] * SCALE
            pygame.draw.line(surf, COLORS[k][0], (tx, self.helipad_y * SCALE), (tx, self.helipad_y * SCALE + 12), 3)
        for x in self.pad:
            y1 = self.helipad_y * SCALE
            pygame.draw.line(surf, (255, 255, 255), (x * SCALE, y1), (x * SCALE, y1 + 50), 1)
            pygame.draw.polygon(surf, (204, 204, 0), [(x * SCALE, y1 + 50), (x * SCALE, y1 + 40), (x * SCALE + 25, y1 + 45)])
        surf = pygame.transform.flip(surf, False, True)
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))
