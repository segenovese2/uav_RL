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
        progress_scale: float = 2.0,
        return_progress_scale: float = 15.0,
        midpoint_bonus: float = 25.0,
        dwell_bonus_per_step: float = 1.0,
        return_bonus: float = 100.0,
        progress_radius: float = 1.5,
    ):
        super(UAVEnv, self).__init__()

        self.grid_size = grid_size
        self.cell_size = 40
        self.window_size = grid_size * self.cell_size
        self.render_mode = render_mode

        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([0, 0])

        # Midpoint computed from actual user positions
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
        self.dwell_bonus_per_step = dwell_bonus_per_step
        self.return_bonus = return_bonus
        self.return_progress_scale = return_progress_scale

        self.action_space = spaces.Discrete(4)
        # 9 dims: original 8 + steps_remaining_ratio
        # phase_flag normalised to [0, 0.5, 1.0] across 3 phases
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

        self.obstacles = self._create_obstacles()
        self.users = self._create_users()
        self.nlos_single, self.nlos_both = self._create_nlos_conditions()

        self.current_pos = None
        self.trajectory = []
        self.steps = 0
        # phase 0: navigate to midpoint
        # phase 1: dwell at midpoint
        # phase 2: return to start
        self.phase = 0
        self._prev_dist = None
        self._dwell_count = 0

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
        self._dwell_count = 0
        self.trajectory = [self.current_pos.copy()]
        self._prev_dist = float(np.linalg.norm(self.current_pos - self.midpoint))
        return self._get_observation(), {}

    def step(self, action: int):
        # Safe coercion -- handles numpy scalars and SAC continuous arrays
        if hasattr(action, '__len__'):
            action = int(np.argmax(action))
        else:
            action = int(action)

        self.steps += 1
        new_pos = self.current_pos.copy()

        if action == 0:   new_pos[1] = min(new_pos[1]+1, self.grid_size-1)
        elif action == 1: new_pos[1] = max(new_pos[1]-1, 0)
        elif action == 2: new_pos[0] = max(new_pos[0]-1, 0)
        elif action == 3: new_pos[0] = min(new_pos[0]+1, self.grid_size-1)

        hit_obstacle = list(new_pos) in self.obstacles

        if hit_obstacle:
            reward = -5.0
            sum_rate = 0.0
        else:
            self.current_pos = new_pos
            self.trajectory.append(self.current_pos.copy())

            # Base comms reward -- identical to original env
            sum_rate, _ = self._calculate_reward()
            reward = sum_rate

            # Additive navigation shaping on top
            reward += self._navigation_shaping()

        truncated = self.steps >= self.max_steps
        if truncated:
            reward += self._calculate_final_reward()

        info = {
            'trajectory': self.trajectory.copy(),
            'steps': self.steps,
            'phase': self.phase,
            'dwell_count': self._dwell_count,
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
        shaping = 0.0
        steps_remaining = self.max_steps - self.steps

        if self.phase == 0:
            # Navigate to midpoint
            current_dist = float(np.linalg.norm(self.current_pos - self.midpoint))
            shaping += self.progress_scale * (self._prev_dist - current_dist)
            self._prev_dist = current_dist

            if self._at_midpoint():
                shaping += self.midpoint_bonus
                self.phase = 1
                self._dwell_count = 0

        elif self.phase == 1:
            # Dwell at midpoint until budget forces return.
            # steps_needed uses 1.5x straight-line distance as path estimate
            # plus a small buffer, ensuring the agent leaves with enough time.
            dist_to_start = float(np.linalg.norm(self.current_pos - self.start_pos))
            steps_needed_to_return = int(math.ceil(dist_to_start * 1.5)) + 3
            must_leave = steps_remaining <= steps_needed_to_return

            # Always dwell for at least 5 steps before must_leave can trigger
            if must_leave and self._dwell_count >= 5:
                self.phase = 2
                self._prev_dist = dist_to_start
            else:
                if self._at_midpoint():
                    self._dwell_count += 1
                    shaping += self.dwell_bonus_per_step
                else:
                    # Drifting resets the counter -- must dwell consecutively
                    self._dwell_count = 0

        elif self.phase == 2:
            # Return to start
            current_dist = float(np.linalg.norm(self.current_pos - self.start_pos))
            shaping += self.return_progress_scale * (self._prev_dist - current_dist)
            self._prev_dist = current_dist

            if self._at_start():
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

        # Target position changes per phase
        if self.phase == 0 or self.phase == 1:
            target = self.midpoint
        else:
            target = self.start_pos

        target_x = target[0] / (self.grid_size - 1)
        target_y = target[1] / (self.grid_size - 1)

        # Normalised phase: 0.0, 0.5, 1.0
        phase_flag = float(self.phase) / 2.0

        # Steps remaining ratio -- critical for dwell phase so agent knows
        # when it needs to leave to make it back in time
        steps_remaining_ratio = (self.max_steps - self.steps) / self.max_steps

        return np.array(
            [uav_x, uav_y, user1_dist, user2_dist, step_ratio,
             target_x, target_y, phase_flag, steps_remaining_ratio],
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
        # Penalise agents that never reach the midpoint -- prevents camping
        # near a user. Only fires if the agent failed -- working agents
        # reach phase 1 or 2 and are unaffected by the phase 0 penalty.
        if self.phase == 0:
            return -200.0
        elif self.phase == 1:
            return -50.0
        else:
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

        mx = int(self.midpoint[0]*self.cell_size + self.cell_size//2)
        my = int((self.grid_size-1-self.midpoint[1])*self.cell_size + self.cell_size//2)
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
        phase_colours = {0: (255,165,0), 1: (0,200,100), 2: (200,0,200)}
        pygame.draw.circle(self.screen, phase_colours[self.phase], (ux, uy), 8)

        steps_remaining = self.max_steps - self.steps
        phase_labels = {
            0: "Phase 0: -> Midpoint",
            1: f"Phase 1: Dwell ({self._dwell_count} steps | {steps_remaining} remaining)",
            2: "Phase 2: -> Start",
        }
        info_surf = self.font.render(
            f"Step {self.steps}/{self.max_steps}  |  {phase_labels[self.phase]}",
            True, (0,0,0)
        )
        self.screen.blit(info_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
