from __future__ import annotations

import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True

try:
    from model.policy import MultiModalSequenceTransformer
except ImportError:
    import torch.nn as nn
    class MultiModalSequenceTransformer(nn.Module):
        def __init__(self, embed_dim=128, num_primitives=10, num_fingers=7, num_heads=4, num_layers=2):
            super().__init__()
            self.embed_dim = embed_dim
            self.vis_cnn = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU()
            )
            self.tac_cnn = nn.Sequential(
                nn.Conv2d(7, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU()
            )
            self.ld_mlp = nn.Sequential(
                nn.Linear(14, 64),
                nn.ReLU()
            )
            self.token_map = nn.Sequential(
                nn.Linear(320, embed_dim),
                nn.ReLU()
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, 10, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.primitive_head = nn.Linear(embed_dim, num_primitives)
            self.finger_head = nn.Linear(embed_dim, num_fingers)
            self.force_head = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()
            )
            self.direction_head = nn.Linear(embed_dim, 2)
            self.value_head = nn.Linear(embed_dim, 1)

        def forward(self, vis_grid, tac_image, low_dim):
            batch_size, seq_len, _, _, _ = vis_grid.shape
            vis_flat = vis_grid.view(-1, 6, 16, 16)
            tac_flat = tac_image.view(-1, 7, 8, 8)
            ld_flat = low_dim.view(-1, 14)
            vis_emb = self.vis_cnn(vis_flat).view(batch_size, seq_len, 128)
            tac_emb = self.tac_cnn(tac_flat).view(batch_size, seq_len, 128)
            ld_emb = self.ld_mlp(ld_flat).view(batch_size, seq_len, 64)
            fused = torch.cat([vis_emb, tac_emb, ld_emb], dim=-1)
            tokens = self.token_map(fused)
            tokens = tokens + self.pos_embed[:, :seq_len, :]
            tokens_out = self.transformer(tokens)
            rep = tokens_out[:, -1, :]
            prim_logits = self.primitive_head(rep)
            finger_logits = self.finger_head(rep)
            force = self.force_head(rep).squeeze(-1)
            direction = self.direction_head(rep)
            dir_norm = torch.norm(direction, p=2, dim=-1, keepdim=True) + 1e-8
            direction = direction / dir_norm
            value = self.value_head(rep).squeeze(-1)
            return prim_logits, finger_logits, force, direction, value

ACTION_PRIMITIVES = ["brace", "stabilize", "push", "drag", "pivot", "roll", "lift_edge", "tap", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._batch_agents = None
        
    def reset(self, task_info):
        if self.model is None:
            self.model = MultiModalSequenceTransformer().to(self.device)
            model_path = Path(__file__).resolve().parent / "model" / "policy.pth"
            if model_path.exists():
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception as exc:
                    print(f"Warning: Failed to load policy weights: {exc}", file=sys.stderr)
            self.model.eval()
            
        self.task_type = task_info["task_type"]
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.tool_need = float(task_info.get("tool_goal", {}).get("progress", 0.0))
        self.obstacles = task_info.get("scene_context", {}).get("obstacles", [])
        self.corridor = task_info.get("scene_context", {}).get("corridor", {})
        
        obj = task_info.get("object", {})
        self.fragile = float(obj.get("fragility", 0.0)) > 0.58
        self.mass = float(obj.get("mass", 1.0))
        self.compliance = float(obj.get("compliance", 0.35))
        self.size = float(obj.get("size", 0.12))
        
        self.prev_raw_pose = None
        self.est_x = 0.0
        self.est_y = 0.0
        self.est_theta = 0.0
        self.est_vx = 0.0
        self.est_vy = 0.0
        self.est_vtheta = 0.0
        
        self.history_low_dim = []
        self.history_vision = []
        self.history_tactile = []
        
        self.tool_progress = 0.0
        self.sequence_progress = 0.0
        self.last_action_taken = None
        self._batch_agents = None

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        if self.model is None:
            self.model = MultiModalSequenceTransformer().to(self.device)
            model_path = Path(__file__).resolve().parent / "model" / "policy.pth"
            if model_path.exists():
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception as exc:
                    print(f"Warning: Failed to load policy weights: {exc}", file=sys.stderr)
            self.model.eval()
            
        for agent, task_info in zip(self._batch_agents, task_infos):
            agent.reset(task_info)
            agent.model = self.model
            agent.device = self.device

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        actions = []
        for obs in observations:
            idx = int(obs.get("batch_index", len(actions)))
            actions.append(self._batch_agents[idx].act(obs))
        return actions

    def wrap_angle(self, val):
        while val > math.pi:
            val -= 2 * math.pi
        while val < -math.pi:
            val += 2 * math.pi
        return val

    def _normalize(self, vec):
        n = math.sqrt(vec[0]**2 + vec[1]**2)
        if n < 1e-9:
            return [0.0, 0.0]
        return [vec[0] / n, vec[1] / n]

    def _cap(self, force, damage_risk):
        if self.fragile and damage_risk > 0.10:
            return min(force, 0.68)
        return force

    def update_tool_progress(self, action, contact):
        if action and action["primitive"] == "tap":
            force = action["force"]
            finger = action["finger"]
            dir_x, dir_y = action["direction"]
            
            dx, dy = self._normalize([dir_x, dir_y])
            ax, ay = self._normalize(self.tool_axis)
            align = max(0.0, ax * dx + ay * dy)
            
            strength = {
                "thumb": 1.08,
                "index": 1.00,
                "middle": 0.95,
                "ring": 0.72,
                "pinky": 0.62,
                "palm": 1.18,
                "wrist": 1.28,
            }.get(finger, 1.0)
            
            if force >= 0.08:
                tap_normal = force * strength * (0.82 + 0.28 * contact)
                self.tool_progress += align * tap_normal * (0.055 + 0.035 * contact)
                if finger in self.reserved:
                    self.sequence_progress += align * tap_normal * 0.065

    def _avoid_obstacles(self, direction, x, y):
        adjust_x = float(direction[0])
        adjust_y = float(direction[1])
        
        for obstacle in self.obstacles:
            try:
                cx = float(obstacle.get("x", 0.0))
                cy = float(obstacle.get("y", 0.0))
                radius = float(obstacle.get("radius", 0.08))
            except Exception:
                continue
                
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            safety_margin = radius + self.size * 0.5 + 0.15
            
            if dist < safety_margin and dist > 1e-8:
                rep_scale = (safety_margin - dist) / safety_margin
                rx, ry = dx / dist, dy / dist
                adjust_x += rx * rep_scale * 1.5
                adjust_y += ry * rep_scale * 1.5
                
                tx, ty = -ry, rx
                dot = adjust_x * tx + adjust_y * ty
                sign = 1.0 if dot >= 0.0 else -1.0
                adjust_x += tx * sign * rep_scale * 1.0
                adjust_y += ty * sign * rep_scale * 1.0
                
        width = float(self.corridor.get("width", 0.0))
        if width > 0.0:
            try:
                axis = self.corridor.get("axis", [1.0, 0.0])
                offset = float(self.corridor.get("offset", 0.0))
                ax, ay = self._normalize(axis)
                
                lateral = -ay * x + ax * y - offset
                half = width * 0.5
                if abs(lateral) > half * 0.3:
                    sign = 1.0 if lateral >= 0.0 else -1.0
                    adjust_x -= (-ay) * sign * 0.5
                    adjust_y -= ax * sign * 0.5
            except Exception:
                pass
                
        return self._normalize([adjust_x, adjust_y])

    def act(self, observation):
        step = int(observation["step"])
        
        if step > 0:
            self.update_tool_progress(self.last_action_taken, float(observation["contact_summary"]["coverage"]))
            
        low_dim = np.array(observation["low_dim_state"]["values"], dtype=np.float32)
        vision = np.array(observation["vision_grid_16x16"]["values"], dtype=np.float32).reshape(6, 16, 16) / 255.0
        tactile = np.array(observation["tactile_image_7x8x8"]["values"], dtype=np.float32).reshape(7, 8, 8) / 255.0
        
        pose = observation["object_pose_estimate"]
        is_dropout = False
        if step > 0 and self.prev_raw_pose is not None:
            if (abs(pose["x"] - self.prev_raw_pose["x"]) < 1e-5 and 
                abs(pose["y"] - self.prev_raw_pose["y"]) < 1e-5 and 
                abs(pose["theta"] - self.prev_raw_pose["theta"]) < 1e-5):
                is_dropout = True
                
        self.prev_raw_pose = dict(pose)
        
        if step == 0:
            self.est_x = pose["x"]
            self.est_y = pose["y"]
            self.est_theta = pose["theta"]
            self.est_vx = 0.0
            self.est_vy = 0.0
            self.est_vtheta = 0.0
        else:
            alpha, beta = 0.6, 0.5
            if is_dropout:
                self.est_x += self.est_vx
                self.est_y += self.est_vy
                self.est_theta = self.wrap_angle(self.est_theta + self.est_vtheta)
                self.est_vx *= 0.90
                self.est_vy *= 0.90
                self.est_vtheta *= 0.90
            else:
                prev_x, prev_y, prev_theta = self.est_x, self.est_y, self.est_theta
                self.est_x = (1 - alpha) * (self.est_x + self.est_vx) + alpha * pose["x"]
                self.est_y = (1 - alpha) * (self.est_y + self.est_vy) + alpha * pose["y"]
                self.est_theta = self.wrap_angle((1 - alpha) * (self.est_theta + self.est_vtheta) + alpha * pose["theta"])
                
                self.est_vx = (1 - beta) * self.est_vx + beta * (self.est_x - prev_x)
                self.est_vy = (1 - beta) * self.est_vy + beta * (self.est_y - prev_y)
                self.est_vtheta = (1 - beta) * self.est_vtheta + beta * self.wrap_angle(self.est_theta - prev_theta)
                
        delay = int(observation["sensor_status"]["action_delay_steps_hint"])
        pred_x = self.est_x + delay * self.est_vx
        pred_y = self.est_y + delay * self.est_vy
        pred_theta = self.wrap_angle(self.est_theta + delay * self.est_vtheta)
        
        target = observation["target_pose"]
        dx = target["x"] - pred_x
        dy = target["y"] - pred_y
        dtheta = self.wrap_angle(target["theta"] - pred_theta)
        dist = math.sqrt(dx*dx + dy*dy)
        
        final_target = observation["final_target_pose"]
        final_dx = final_target["x"] - pred_x
        final_dy = final_target["y"] - pred_y
        final_dtheta = self.wrap_angle(final_target["theta"] - pred_theta)
        final_dist = math.sqrt(final_dx*final_dx + final_dy*final_dy)
        
        self.history_low_dim.append(low_dim)
        self.history_vision.append(vision)
        self.history_tactile.append(tactile)
        
        self.history_low_dim = self.history_low_dim[-6:]
        self.history_vision = self.history_vision[-6:]
        self.history_tactile = self.history_tactile[-6:]
        
        pad_len = 6 - len(self.history_low_dim)
        if pad_len > 0:
            pad_ld = [np.zeros(14, dtype=np.float32) for _ in range(pad_len)]
            pad_vis = [np.zeros((6, 16, 16), dtype=np.float32) for _ in range(pad_len)]
            pad_tac = [np.zeros((7, 8, 8), dtype=np.float32) for _ in range(pad_len)]
            
            in_ld = np.stack(pad_ld + self.history_low_dim)
            in_vis = np.stack(pad_vis + self.history_vision)
            in_tac = np.stack(pad_tac + self.history_tactile)
        else:
            in_ld = np.stack(self.history_low_dim)
            in_vis = np.stack(self.history_vision)
            in_tac = np.stack(self.history_tactile)
            
        t_ld = torch.tensor(in_ld, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_vis = torch.tensor(in_vis, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_tac = torch.tensor(in_tac, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prim_logits, finger_logits, force_pred, dir_pred, _ = self.model(t_vis, t_tac, t_ld)
            
        prim_idx = torch.argmax(prim_logits, dim=-1).item()
        finger_idx = torch.argmax(finger_logits, dim=-1).item()
        force = force_pred.item()
        direction = dir_pred.squeeze(0).cpu().numpy().tolist()
        
        pred_primitive = ACTION_PRIMITIVES[prim_idx]
        pred_finger = FINGERS[finger_idx]
        
        action = {
            "primitive": pred_primitive,
            "finger": pred_finger,
            "force": force,
            "direction": direction
        }
        
        contact_coverage = float(observation["contact_summary"]["coverage"])
        slip_risk = float(observation["contact_summary"]["slip_risk"])
        damage_risk = float(observation["contact_summary"]["damage_risk"])
        
        if step < 1 or contact_coverage < 0.35 or slip_risk > 0.88 or damage_risk > 0.95:
            action = {
                "primitive": "brace",
                "finger": "palm",
                "force": 0.30,
                "direction": [0.0, 0.0]
            }
            
        elif self.task_type in {"tool_use", "resource_sequence"} and self.tool_progress < self.tool_need * 0.98:
            ax, ay = self._normalize(self.tool_axis)
            
            path_blocked = False
            blocked_obs = None
            for obs in self.obstacles:
                ox = obs["x"] - pred_x
                oy = obs["y"] - pred_y
                odist = math.sqrt(ox*ox + oy*oy)
                if odist < obs["radius"] + self.size * 0.5 + 0.16:
                    dot = ox * ax + oy * ay
                    if dot > 0:
                        path_blocked = True
                        blocked_obs = obs
                        break
                        
            if path_blocked:
                ox = blocked_obs["x"] - pred_x
                oy = blocked_obs["y"] - pred_y
                push_dir = self._normalize([-ox, -oy])
                non_reserved = [f for f in ["palm", "thumb", "pinky", "ring", "middle", "index"] if f not in self.reserved]
                finger = non_reserved[0] if non_reserved else "palm"
                action = {
                    "primitive": "push",
                    "finger": finger,
                    "force": self._cap(0.65, damage_risk),
                    "direction": push_dir
                }
            else:
                finger = self.reserved[0] if self.task_type == "resource_sequence" and self.reserved else "index"
                action = {
                    "primitive": "tap",
                    "finger": finger,
                    "force": self._cap(0.85, damage_risk),
                    "direction": [ax, ay]
                }
        
        else:
            if self.task_type == "resource_sequence" and self.tool_progress < 0.04 and action["finger"] in self.reserved:
                non_reserved = [f for f in ["palm", "thumb", "pinky", "ring", "middle", "index"] if f not in self.reserved]
                action["finger"] = non_reserved[0] if non_reserved else "palm"
                
            if action["primitive"] == "pivot":
                action["direction"] = [1.0 if dtheta >= 0.0 else -1.0, 0.0]
            elif action["primitive"] in {"push", "drag", "roll"}:
                target_dir = self._normalize([dx, dy])
                blended = self._normalize([
                    0.80 * target_dir[0] + 0.20 * action["direction"][0],
                    0.80 * target_dir[1] + 0.20 * action["direction"][1]
                ])
                action["direction"] = self._avoid_obstacles(blended, pred_x, pred_y)
                
        if final_dist < 0.080 and abs(final_dtheta) < 0.30:
            if self.task_type not in {"tool_use", "resource_sequence"} or self.tool_progress >= self.tool_need * 0.95:
                action = {
                    "primitive": "finish",
                    "finger": "palm",
                    "force": 0.0,
                    "direction": [0.0, 0.0]
                }
        elif action["primitive"] == "finish":
            target_dir = self._normalize([dx, dy])
            action = {
                "primitive": "push",
                "finger": "palm",
                "force": self._cap(0.70, damage_risk),
                "direction": self._avoid_obstacles(target_dir, pred_x, pred_y)
            }
            
        action["force"] = min(action["force"], self.max_peak_force * 0.9)
        
        self.last_action_taken = action
        return action
