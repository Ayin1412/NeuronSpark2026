import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulator"))

from dexsim_core import DexSimEnv, sanitize_task_info, score_rollout

ACTION_PRIMITIVES = ["brace", "stabilize", "push", "drag", "pivot", "roll", "lift_edge", "tap", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]

def load_tasks(task_file):
    tasks = {}
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                task = json.loads(line)
                tasks[task["id"]] = task
    return tasks

def main():
    train_tasks_path = ROOT / "tasks" / "train_tasks.jsonl"
    rollouts_path = ROOT / "demonstrations" / "weak_train_rollouts.jsonl"
    output_path = ROOT / "reconstructed_dataset.npz"
    
    print("Loading training tasks...")
    tasks = load_tasks(train_tasks_path)
    print(f"Loaded {len(tasks)} tasks.")
    
    print("Replaying demonstrations...")
    
    low_dim_states = []
    vision_grids = []
    tactile_images = []
    prim_labels = []
    finger_labels = []
    forces = []
    directions = []
    returns = []
    episode_lengths = []
    
    with open(rollouts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Replaying episodes"):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        task_id = data["task_id"]
        variant_id = data["variant_id"]
        steps = data["steps"]
        final_score = float(data["metrics"]["score01"])
        
        task = tasks.get(task_id)
        if not task:
            continue
            
        variant = None
        for v in task.get("public_variants", []):
            if v.get("variant_id") == variant_id:
                variant = v
                break
        if not variant:
            for v in task.get("private_rollouts", []):
                if v.get("variant_id") == variant_id:
                    variant = v
                    break
        
        env = DexSimEnv(task, variant)
        
        ep_low_dim = []
        ep_vision = []
        ep_tactile = []
        ep_prim = []
        ep_finger = []
        ep_force = []
        ep_dir = []
        
        for step_data in steps:
            action = step_data["action"]
            obs = env.observation()
            
            low_dim = np.array(obs["low_dim_state"]["values"], dtype=np.float32)
            vision = np.array(obs["vision_grid_16x16"]["values"], dtype=np.uint8).reshape(6, 16, 16)
            tactile = np.array(obs["tactile_image_7x8x8"]["values"], dtype=np.uint8).reshape(7, 8, 8)
            
            prim = ACTION_PRIMITIVES.index(action["primitive"])
            finger = FINGERS.index(action["finger"])
            force = float(action["force"])
            direction = np.array(action["direction"], dtype=np.float32)
            
            ep_low_dim.append(low_dim)
            ep_vision.append(vision)
            ep_tactile.append(tactile)
            ep_prim.append(prim)
            ep_finger.append(finger)
            ep_force.append(force)
            ep_dir.append(direction)
            
            env.step(action)
            
        episode_lengths.append(len(ep_prim))
        low_dim_states.append(np.stack(ep_low_dim))
        vision_grids.append(np.stack(ep_vision))
        tactile_images.append(np.stack(ep_tactile))
        prim_labels.append(np.array(ep_prim, dtype=np.int64))
        finger_labels.append(np.array(ep_finger, dtype=np.int64))
        forces.append(np.array(ep_force, dtype=np.float32))
        directions.append(np.stack(ep_dir))
        returns.append(np.full(len(ep_prim), final_score, dtype=np.float32))

    low_dim_states = np.concatenate(low_dim_states, axis=0)
    vision_grids = np.concatenate(vision_grids, axis=0)
    tactile_images = np.concatenate(tactile_images, axis=0)
    prim_labels = np.concatenate(prim_labels, axis=0)
    finger_labels = np.concatenate(finger_labels, axis=0)
    forces = np.concatenate(forces, axis=0)
    directions = np.concatenate(directions, axis=0)
    returns = np.concatenate(returns, axis=0)
    episode_lengths = np.array(episode_lengths, dtype=np.int32)
    
    print(f"Total transition steps: {len(prim_labels)}")
    print(f"Saving dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        low_dim_states=low_dim_states,
        vision_grids=vision_grids,
        tactile_images=tactile_images,
        prim_labels=prim_labels,
        finger_labels=finger_labels,
        forces=forces,
        directions=directions,
        returns=returns,
        episode_lengths=episode_lengths
    )
    print("完成")

if __name__ == "__main__":
    main()
