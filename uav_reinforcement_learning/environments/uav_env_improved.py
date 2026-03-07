import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Optional

class UAVEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        grid_size=15,
        render_mode=None,
        P: float = 1.0,
        N: float = 1e-3,
        alpha: float = 2.0,
        beta_shadow: float = 0.01,
        stochastic_fading: bool = True,
        uav_height: float = 5.0,
        midpoint: Optional[np.ndarray] = None,
        # --- Shaping params, intentionally kept in the same scale as sum_rate ---
        # sum_rate typically ranges 5-9 bits/s/Hz in this environment.
        # progress_scale: reward per grid-cell of distance closed toward target.
        #   ~0.4 means closing 1 cell ~= 5% of a typical comms step reward.
        #   Large enough to guide, small enough not to override comms quality.
        progress_scale: float = 0.4,
        # return_progress_scale: MUCH higher during return phase to force going home.
        #   Without this, agent perfectly content oscillating at good comms spot.
        #   ~2.0 guides home slowly while still valuing comms rewards during return.
        return_progress_scale: float = 5.0,
        # midpoint_bonus: one-time reward for reaching the midpoint.
        #   Sized as roughly one strong comms step (~8 bits/s/Hz).
        midpoint_bonus: float = 25.0,
        # return_bonus: one-time reward for completing the return leg.
        #   MUCH larger to incentivize returning home within 50-step limit.
        return_bonus: float = 100.0,
        # How close (grid cells) counts as "reached" for milestone triggers.
        progress_radius: float = 1.5,
    ):
        super(UAVEnv, self).__init__()

        self.grid_size = grid_size
        self.cell_size = 40
        self.window_size = grid_size * self.cell_size
        self.render_mode = render_mode

        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([0, 0])

        users = self._create_users()
        self.midpoint = np.array(midpoint) if midpoint is not None else np.array([
            (users[0][0] + users[1][0]) / 2,
            (users[0][1] + users[1][1]) / 2
        ])
        self.progress_radius = progress_radius

        self.max_steps = 50

        self.P = P
        self.N = N
        self.alpha = alpha
        self.beta_shadow = beta_shadow
        self.stochastic_fading = stochastic_fading
        self.uav_height = uav_height

        self.progress_scale = progress_scale
        self.midpoint_bonus = midpoint_bonus
        self.return_bonus = return_bonus
        self.return_progress_scale = return_progress_scale

        self.action_space = spaces.Discrete(4)
        # Observation gains 3 new dims over original: target_x, target_y, phase flag
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        self.obstacles = self._create_obstacles()
        self.users = self._create_users()
        self.nlos_single, self.nlos_both = self._create_nlos_conditions()

        self.current_pos = None
        self.trajectory = []
        self.steps = 0
        self.phase = 0
        self._prev_dist = None

        self.screen = None
        self.clock = None
        self.font = None
        if render_mode == "human":
            self._init_pygame()

        self.max_possible_sum_rate = self._compute_max_sum_rate()

    # ------------------------------------------------------------------ #
    #  WORLD GENERATION (unchanged)                                        #
    # ------------------------------------------------------------------ #
    def _create_obstacles(self):
        return [
            [9,3],[9,4],[9,5],[9,6],
            [10,3],[10,4],[10,5],[10,6]
        ]

    def _create_users(self):
        return [[4,12], [12,8]]

    def _create_nlos_conditions(self):
        nlos_single = []
        nlos_both = []
        user1 = np.array(self.users[0])
        user2 = np.array(self.users[1])
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if [x,y] in self.obstacles:
                    continue
                pos = np.array([x,y])
                los1 = self._has_los(pos, user1)
                los2 = self._has_los(pos, user2)
                if not los1 and not los2:
                    nlos_both.append([x,y])
                elif not los1 or not los2:
                    nlos_single.append([x,y])
        return nlos_single, nlos_both

    def _has_los(self, pos1, pos2):
        x1, y1 = int(pos1[0]), int(pos1[1])
        x2, y2 = int(pos2[0]), int(pos2[1])
        dx = abs(x2-x1); dy = abs(y2-y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2>x1 else -1
        y_inc = 1 if y2>y1 else -1
        err = dx - dy
        dx *= 2; dy *= 2
        for _ in range(n):
            if not ((x==x1 and y==y1) or (x==x2 and y==y2)):
                if [x,y] in self.obstacles:
                    return False
            if err > 0:
                x += x_inc; err -= dy
            else:
                y += y_inc; err += dx
        return True

    # ------------------------------------------------------------------ #
    #  RESET / STEP                                                        #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        self.steps = 0
        self.phase = 0
        self.trajectory = [self.current_pos.copy()]
        self._prev_dist = float(np.linalg.norm(self.current_pos - self.midpoint))
        return self._get_observation(), {}

    def step(self, action: int):
        self.steps += 1
        new_pos = self.current_pos.copy()

        if action == 0:   new_pos[1] = min(new_pos[1]+1, self.grid_size-1)
        elif action == 1: new_pos[1] = max(new_pos[1]-1, 0)
        elif action == 2: new_pos[0] = max(new_pos[0]-1, 0)
        elif action == 3: new_pos[0] = min(new_pos[0]+1, self.grid_size-1)

        hit_obstacle = list(new_pos) in self.obstacles

        if hit_obstacle:
            # Unchanged from original
            reward = -5.0
            sum_rate = 0.0
        else:
            self.current_pos = new_pos
            self.trajectory.append(self.current_pos.copy())

            # Original comms reward, completely unchanged
            sum_rate, _ = self._calculate_reward()
            reward = sum_rate

            # Additive navigation shaping on top, same reward scale
            reward += self._navigation_shaping()

        truncated = self.steps >= self.max_steps
        if truncated:
            reward += self._calculate_final_reward()

        info = {
            'trajectory': self.trajectory.copy(),
            'steps': self.steps,
            'phase': self.phase,
            'hit_obstacle': hit_obstacle,
            'communication_quality': self._get_communication_quality(),
            'at_midpoint': self._at_midpoint(),
            'at_start': self._at_start(),
            'sum_rate': sum_rate if not hit_obstacle else 0.0,
        }

        return self._get_observation(), float(reward), False, truncated, info

    # ------------------------------------------------------------------ #
    #  NAVIGATION SHAPING                                                  #
    # ------------------------------------------------------------------ #
    def _navigation_shaping(self) -> float:
        """
        Additive reward on top of raw sum_rate, kept in the same scale.

        Per-step progress: progress_scale * (cells closed toward target)
          - 1 cell closer  -> +0.4  (~5% of a typical comms step)
          - 1 cell further -> -0.4  (mild deterrent, won't dominate comms)
        
        During RETURN phase: use moderate progress scale to guide home without dominating.
          - 1 cell closer to start -> +2.0  (guides home but comms rewards still matter)
          - Agent stays at midpoint collecting comms, then gradually returns near step 50.

        Milestones are sized as 1-2 strong comms steps so the agent
        notices them without them swamping the comms learning signal.
        """
        shaping = 0.0

        target = self.midpoint if self.phase == 0 else self.start_pos
        current_dist = float(np.linalg.norm(self.current_pos - target))

        # Use aggressive progress scale during return phase
        scale = self.progress_scale if self.phase == 0 else self.return_progress_scale
        shaping += scale * (self._prev_dist - current_dist)
        self._prev_dist = current_dist

        if self.phase == 0 and self._at_midpoint():
            shaping += self.midpoint_bonus
            self.phase = 1
            self._prev_dist = float(np.linalg.norm(self.current_pos - self.start_pos))

        elif self.phase == 1 and self._at_start():
            shaping += self.return_bonus

        return shaping

    def _at_midpoint(self) -> bool:
        return float(np.linalg.norm(self.current_pos - self.midpoint)) <= self.progress_radius

    def _at_start(self) -> bool:
        return float(np.linalg.norm(self.current_pos - self.start_pos)) <= self.progress_radius

    # ------------------------------------------------------------------ #
    #  OBSERVATION                                                         #
    # ------------------------------------------------------------------ #
    def _get_observation(self):
        uav_x = self.current_pos[0] / (self.grid_size - 1)
        uav_y = self.current_pos[1] / (self.grid_size - 1)

        user1_dist = np.linalg.norm(self.current_pos - np.array(self.users[0])) / (self.grid_size * 1.5)
        user2_dist = np.linalg.norm(self.current_pos - np.array(self.users[1])) / (self.grid_size * 1.5)
        step_ratio = self.steps / self.max_steps

        # Phase target exposed to policy so it can condition on which leg it's on
        target = self.midpoint if self.phase == 0 else self.start_pos
        target_x = target[0] / (self.grid_size - 1)
        target_y = target[1] / (self.grid_size - 1)
        phase_flag = float(self.phase)  # 0.0 or 1.0

        return np.array(
            [uav_x, uav_y, user1_dist, user2_dist, step_ratio,
             target_x, target_y, phase_flag],
            dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    #  COMMS REWARD (unchanged from original)                              #
    # ------------------------------------------------------------------ #
    def _calculate_reward(self):
        pos = self.current_pos
        sum_rate = 0.0
        for user in self.users:
            horiz_dist = np.linalg.norm(pos - np.array(user))
            d_k = math.sqrt(horiz_dist**2 + self.uav_height**2)
            L_dist = d_k**(-self.alpha)
            los = self._has_los(pos, np.array(user))
            beta = 1.0 if los else self.beta_shadow
            if self.stochastic_fading:
                x_ray = np.random.rayleigh(scale=1.0)
                fading = 10 ** (x_ray / 10.0)
            else:
                fading = 1.0
            L_k = L_dist * fading * beta
            R_k = math.log2(1 + (self.P / self.N) * L_k)
            sum_rate += R_k
        return float(sum_rate), False

    def _calculate_final_reward(self):
        return 0.0

    # ------------------------------------------------------------------ #
    #  COMM QUALITY HELPERS (unchanged)                                    #
    # ------------------------------------------------------------------ #
    def _compute_sum_rate_at_pos(self, pos, deterministic=True):
        sum_rate = 0.0
        for user in self.users:
            horiz_dist = np.linalg.norm(pos - np.array(user))
            d_k = math.sqrt(horiz_dist**2 + self.uav_height**2)
            L_dist = d_k**(-self.alpha)
            los = self._has_los(pos, np.array(user))
            beta = 1.0 if los else self.beta_shadow
            L_k = L_dist * 1.0 * beta
            R_k = math.log2(1 + (self.P / self.N) * L_k)
            sum_rate += R_k
        return sum_rate

    def _compute_max_sum_rate(self):
        best = 1e-9
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                sr = self._compute_sum_rate_at_pos(np.array([x, y]))
                best = max(best, sr)
        return float(best)

    def _get_communication_quality(self):
        return float(min(1.0, self._compute_sum_rate_at_pos(self.current_pos) / self.max_possible_sum_rate))

    # ------------------------------------------------------------------ #
    #  RENDERING                                                           #
    # ------------------------------------------------------------------ #
    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("UAV Trajectory")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            self._init_pygame()

        self.screen.fill((255, 255, 255))

        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200,200,200),
                             (x*self.cell_size, 0), (x*self.cell_size, self.window_size))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200,200,200),
                             (0, y*self.cell_size), (self.window_size, y*self.cell_size))

        for cell in self.nlos_both:
            rect = pygame.Rect(cell[0]*self.cell_size, (self.grid_size-1-cell[1])*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (120,120,120), rect)
        for cell in self.nlos_single:
            rect = pygame.Rect(cell[0]*self.cell_size, (self.grid_size-1-cell[1])*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (180,180,180), rect)

        for obs in self.obstacles:
            rect = pygame.Rect(obs[0]*self.cell_size, (self.grid_size-1-obs[1])*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (80,80,80), rect)

        mx = self.midpoint[0]*self.cell_size + self.cell_size//2
        my = (self.grid_size-1-self.midpoint[1])*self.cell_size + self.cell_size//2
        pygame.draw.circle(self.screen, (0,200,100), (mx, my), 8, 3)
        mid_label = self.font.render("M", True, (0,150,80))
        self.screen.blit(mid_label, mid_label.get_rect(center=(mx, my)))

        for user in self.users:
            cx = user[0]*self.cell_size + self.cell_size//2
            cy = (self.grid_size-1-user[1])*self.cell_size + self.cell_size//2
            pygame.draw.circle(self.screen, (0,100,255), (cx, cy), 10)
            label = self.font.render("UE", True, (255,255,255))
            self.screen.blit(label, label.get_rect(center=(cx, cy)))

        if len(self.trajectory) > 1:
            pts = [(p[0]*self.cell_size + self.cell_size//2,
                    (self.grid_size-1-p[1])*self.cell_size + self.cell_size//2)
                   for p in self.trajectory]
            pygame.draw.lines(self.screen, (0,0,0), False, pts, 3)

        ux = self.current_pos[0]*self.cell_size + self.cell_size//2
        uy = (self.grid_size-1-self.current_pos[1])*self.cell_size + self.cell_size//2
        uav_colour = (255,165,0) if self.phase == 0 else (200,0,200)
        pygame.draw.circle(self.screen, uav_colour, (ux, uy), 8)

        phase_str = "Phase 0: -> Midpoint" if self.phase == 0 else "Phase 1: -> Start"
        info_surf = self.font.render(f"Step {self.steps}/{self.max_steps}  |  {phase_str}", True, (0,0,0))
        self.screen.blit(info_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)