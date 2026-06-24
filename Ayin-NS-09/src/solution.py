from __future__ import annotations
import json
import argparse
from collections import Counter
from typing import Optional, Tuple, Dict, List, Set

GRID_SIZE = 12
EVENT_KEYS = ['goal_reached', 'collision', 'hazard', 'box_on_goal', 'key_collected', 'portal_used']
EVENT_ORDER_WIDTH = 3
TERMINALS = ['goal', 'hazard', 'blocked', 'active', 'timeout']
CONVEYOR_DELTA = {'<': (-1, 0), '>': (1, 0), '^': (0, -1), 'v': (0, 1)}
ACTION_DELTA = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0), 'WAIT': (0, 0)}

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_pos(grid, char):
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == char:
                return (c, r)
    return None

def get_all_pos(grid, char):
    return [(c, r) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == char]

def classify_time(step, total):
    """Classify step into early/mid/late bucket."""
    if total == 0:
        return 'never'
    t = (step + 1) / total
    if t <= 1/3:
        return 'early'
    elif t <= 2/3:
        return 'mid'
    else:
        return 'late'

def infer_context_profile(context_episodes: List[dict], init_grid: List[str]) -> dict:
    profile = {}
    
    o_finals = [get_pos(ep['observed_final_full_grid'], 'O') for ep in context_episodes]
    o_finals_valid = [p for p in o_finals if p is not None]
    if o_finals_valid:
        pos_count = Counter(str(p) for p in o_finals_valid)
        majority_str = pos_count.most_common(1)[0][0]
        profile['o_equilibrium'] = eval(majority_str)
    else:
        profile['o_equilibrium'] = get_pos(init_grid, 'O')
    
    b_finals = [get_pos(ep['observed_final_full_grid'], 'B') for ep in context_episodes]
    b_finals_valid = [p for p in b_finals if p is not None]
    if b_finals_valid:
        pos_count = Counter(str(p) for p in b_finals_valid)
        majority_str = pos_count.most_common(1)[0][0]
        profile['b_equilibrium'] = eval(majority_str)
    else:
        profile['b_equilibrium'] = get_pos(init_grid, 'B')
    profile['k_start'] = get_pos(init_grid, 'K')
    profile['probe_events'] = [ep['observed_final_events'] for ep in context_episodes]
    profile['probe_terminals'] = [ep['observed_final_terminal'] for ep in context_episodes]
    profile['probe_grids'] = [ep['observed_final_full_grid'] for ep in context_episodes]
    profile['probe_actions'] = [ep['actions'] for ep in context_episodes]
    
    return profile


class WorldState:
    """Mutable world state for simulation."""
    
    def __init__(self, init_grid: List[str], profile: dict):
        self.H = len(init_grid)
        self.W = len(init_grid[0]) if init_grid else 0
        self.init_grid = init_grid
        self.profile = profile
        self.smap = [list(row) for row in init_grid]
        self.entity_original_positions = {}
        for entity_char in 'ABOK':
            pos = get_pos(init_grid, entity_char)
            if pos:
                x, y = pos
                self.entity_original_positions[entity_char] = pos
                self.smap[y][x] = '.'
        
        self.goals = set(get_all_pos(init_grid, 'G'))
        self.hazards = set(get_all_pos(init_grid, 'H'))
        self.portals = get_all_pos(init_grid, 'P')
        self.ice_cells = set(get_all_pos(init_grid, 'I'))
        self.conveyor_map = {(c, r): ch for r, row in enumerate(init_grid) 
                            for c, ch in enumerate(row) if ch in '<>^v'}
        self.doors = set(get_all_pos(init_grid, 'D'))
        
        self.agent_pos = get_pos(init_grid, 'A')
        self.box_pos = profile.get('b_equilibrium') or get_pos(init_grid, 'B')
        self.orb_pos = profile.get('o_equilibrium') or get_pos(init_grid, 'O')
        self.key_pos = profile.get('k_start') or get_pos(init_grid, 'K')
        self.cleared_positions = set(self.entity_original_positions.values())
        
        self.has_key = False
        self.events = {k: False for k in EVENT_KEYS}
        self.event_steps = {}
    
    def cell(self, x: int, y: int) -> str:
        if 0 <= x < self.W and 0 <= y < self.H:
            return self.smap[y][x]
        return '#'
    
    def is_solid(self, x: int, y: int) -> bool:
        c = self.cell(x, y)
        return c == '#' or (c == 'D' and not self.has_key)
    
    def record_event(self, name: str, step: int):
        if not self.events[name]:
            self.events[name] = True
            self.event_steps[name] = step
    
    def slide_on_ice(self, x: int, y: int, dx: int, dy: int, 
                     blockers: Set = None) -> Tuple[int, int, bool]:
        blockers = blockers or set()
        while True:
            nx, ny = x + dx, y + dy
            if self.is_solid(nx, ny):
                return x, y, True
            if (nx, ny) in blockers:
                return x, y, False
            x, y = nx, ny
            if (x, y) not in self.ice_cells:
                return x, y, False
    
    def check_position_events(self, x: int, y: int, step: int):
        if (x, y) in self.goals:
            self.record_event('goal_reached', step)
        
        if (x, y) in self.hazards:
            self.record_event('hazard', step)
        
        if self.key_pos and (x, y) == self.key_pos:
            self.record_event('key_collected', step)
            self.has_key = True
            self.key_pos = None
        
        if (x, y) in self.portals:
            self.record_event('portal_used', step)
            other_portals = [p for p in self.portals if p != (x, y)]
            if other_portals:
                self.agent_pos = other_portals[0]
                tx, ty = self.agent_pos
                if (tx, ty) in self.goals:
                    self.record_event('goal_reached', step)
                if (tx, ty) in self.hazards:
                    self.record_event('hazard', step)
    
    def apply_conveyor(self, pos: Tuple, step: int, 
                       other_blockers: Set = None, is_agent: bool = False) -> Tuple:
        x, y = pos
        if (x, y) not in self.conveyor_map:
            return pos
        
        cdx, cdy = CONVEYOR_DELTA[self.conveyor_map[(x, y)]]
        nx, ny = x + cdx, y + cdy
        blockers = other_blockers or set()
        
        if not self.is_solid(nx, ny) and (nx, ny) not in blockers:
            if is_agent:
                self.check_position_events(nx, ny, step)
            elif (nx, ny) in self.goals:
                self.record_event('box_on_goal', step)
            return (nx, ny)
        return pos
    
    def step(self, action: str, step_idx: int):
        ax, ay = self.agent_pos
        dx, dy = ACTION_DELTA[action]
        
        if action != 'WAIT':
            nx, ny = ax + dx, ay + dy
            dest_solid = self.is_solid(nx, ny)
            dest_is_box = self.box_pos is not None and (nx, ny) == self.box_pos
            dest_is_orb = self.orb_pos is not None and (nx, ny) == self.orb_pos
            
            if dest_solid:
                self.record_event('collision', step_idx)
                
            elif dest_is_box:
                bnx, bny = nx + dx, ny + dy
                b_blockers = {self.agent_pos}
                if self.orb_pos:
                    b_blockers.add(self.orb_pos)
                
                if not self.is_solid(bnx, bny) and (bnx, bny) not in b_blockers:
                    if (bnx, bny) in self.ice_cells:
                        new_bx, new_by, _ = self.slide_on_ice(
                            bnx, bny, dx, dy,
                            b_blockers | ({self.orb_pos} if self.orb_pos else set()))
                        self.box_pos = (new_bx, new_by)
                    else:
                        self.box_pos = (bnx, bny)
                    
                    if self.box_pos in self.goals:
                        self.record_event('box_on_goal', step_idx)
                    self.agent_pos = (nx, ny)
                else:
                    self.record_event('collision', step_idx)
                    
            elif dest_is_orb:
                onx, ony = nx + dx, ny + dy
                o_blockers = {self.agent_pos}
                if self.box_pos:
                    o_blockers.add(self.box_pos)
                
                if not self.is_solid(onx, ony) and (onx, ony) not in o_blockers:
                    if (onx, ony) in self.ice_cells:
                        new_ox, new_oy, _ = self.slide_on_ice(
                            onx, ony, dx, dy,
                            o_blockers | ({self.box_pos} if self.box_pos else set()))
                        self.orb_pos = (new_ox, new_oy)
                    else:
                        self.orb_pos = (onx, ony)
                    self.agent_pos = (nx, ny)
                else:
                    self.record_event('collision', step_idx)
                    
            else:
                self.agent_pos = (nx, ny)
                ax2, ay2 = self.agent_pos
                
                if (ax2, ay2) in self.ice_cells:
                    blockers = set()
                    if self.box_pos: blockers.add(self.box_pos)
                    if self.orb_pos: blockers.add(self.orb_pos)
                    new_ax, new_ay, hit_wall = self.slide_on_ice(ax2, ay2, dx, dy, blockers)
                    if hit_wall:
                        self.record_event('collision', step_idx)
                    self.agent_pos = (new_ax, new_ay)
        
        self.check_position_events(*self.agent_pos, step_idx)
    
    def get_terminal(self) -> str:
        if self.events['goal_reached']:
            return 'goal'
        elif self.events['hazard']:
            return 'hazard'
        elif self.events['collision']:
            return 'blocked'
        return 'timeout'
    
    def get_event_timeline(self, total_steps: int) -> dict:
        return {k: (classify_time(self.event_steps[k], total_steps) 
                    if k in self.event_steps else 'never')
                for k in EVENT_KEYS}
    
    def get_event_order(self) -> list:
        ordered = sorted(self.event_steps.items(), key=lambda x: x[1])
        order = [e[0] for e in ordered[:3]]
        while len(order) < 3:
            order.append('none')
        return order
    
    def build_final_grid(self) -> List[str]:
        grid = [list(row) for row in self.smap]
        
        for dx, dy in self.doors:
            grid[dy][dx] = 'D'
        
        if self.key_pos:
            kx, ky = self.key_pos
            if 0 <= kx < self.W and 0 <= ky < self.H:
                grid[ky][kx] = 'K'
        
        if self.orb_pos is not None:
            ox, oy = self.orb_pos
            if 0 <= ox < self.W and 0 <= oy < self.H:
                grid[oy][ox] = 'O'
        
        if self.box_pos is not None:
            bx, by = self.box_pos
            if 0 <= bx < self.W and 0 <= by < self.H:
                grid[by][bx] = 'B'
        
        if self.agent_pos:
            ax, ay = self.agent_pos
            if 0 <= ax < self.W and 0 <= ay < self.H:
                grid[ay][ax] = 'A'
        
        return [''.join(row) for row in grid]


def simulate_episode(init_grid: List[str], actions: List[str], 
                     profile: dict) -> Tuple[List[str], dict, dict, list, str]:
    state = WorldState(init_grid, profile)
    N = len(actions)
    
    for i, action in enumerate(actions):
        state.step(action, i)
    
    terminal = state.get_terminal()
    events = state.events.copy()
    event_timeline = state.get_event_timeline(N)
    event_order = state.get_event_order()
    final_grid = state.build_final_grid()
    
    return final_grid, events, event_timeline, event_order, terminal


def predict_context(context: dict) -> List[dict]:
    init_grid = context['initial_full_grid']
    context_episodes = context['context_episodes']
    queries = context['queries']
    
    profile = infer_context_profile(context_episodes, init_grid)
    
    predictions = []
    for query in queries:
        q_id = query['query_id']
        q_actions = query['future_actions']
        
        final_grid, events, event_timeline, event_order, terminal = simulate_episode(
            init_grid, q_actions, profile
        )
        
        predictions.append({
            'id': q_id,
            'final_grid': final_grid,
            'events': events,
            'event_timeline': event_timeline,
            'event_order': event_order,
            'terminal': terminal,
        })
    
    return predictions


def evaluate(pred_results: List[dict], ground_truth: List[dict]) -> dict:
    gt_by_id = {item['query_id']: item['label'] for item in ground_truth}
    
    metrics = {
        'total': 0,
        'terminal_correct': 0,
        'event_correct': 0,
        'grid_correct': 0,
        'timeline_correct': 0,
        'order_correct': 0,
    }
    
    for pred in pred_results:
        qid = pred['id']
        if qid not in gt_by_id:
            continue
        label = gt_by_id[qid]
        metrics['total'] += 1
        
        if pred['terminal'] == label['terminal']:
            metrics['terminal_correct'] += 1
        if pred['events'] == label['events']:
            metrics['event_correct'] += 1
        if pred['final_grid'] == label['final_grid']:
            metrics['grid_correct'] += 1
        if pred['event_timeline'] == label['event_timeline']:
            metrics['timeline_correct'] += 1
        if pred['event_order'] == label['event_order']:
            metrics['order_correct'] += 1
    
    total = metrics['total']
    if total > 0:
        metrics['terminal_acc'] = metrics['terminal_correct'] / total
        metrics['event_acc'] = metrics['event_correct'] / total
        metrics['grid_acc'] = metrics['grid_correct'] / total
        metrics['timeline_acc'] = metrics['timeline_correct'] / total
        metrics['order_acc'] = metrics['order_correct'] / total
    
    return metrics


def compute_score(metrics: dict) -> float:
    score = 100
    score += 225 * metrics.get('grid_acc', 0)
    score += 225 * metrics.get('grid_acc', 0)
    score += 250 * metrics.get('event_acc', 0)
    score += 250 * metrics.get('timeline_acc', 0)
    score += 250 * metrics.get('order_acc', 0)
    score += 200 * metrics.get('terminal_acc', 0)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train.jsonl')
    parser.add_argument('--valid', default='valid.jsonl')
    parser.add_argument('--test', default='test.jsonl')
    parser.add_argument('--out', default='results.json')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--eval_valid', action='store_true')
    args = parser.parse_args()
    
    if args.eval_train:
        print("Evaluating on training set...")
        all_preds = []
        all_gt = []
        
        for i, context in enumerate(read_jsonl(args.train)):
            preds = predict_context(context)
            all_preds.extend(preds)
            for q in context['queries']:
                all_gt.append(q)
        
        metrics = evaluate(all_preds, all_gt)
        score = compute_score(metrics)
        print(f"Train metrics:")
        print(f"  Total: {metrics['total']}")
        print(f"  Terminal: {metrics['terminal_acc']:.3f}")
        print(f"  Events: {metrics['event_acc']:.3f}")
        print(f"  Grid: {metrics['grid_acc']:.3f}")
        print(f"  Timeline: {metrics['timeline_acc']:.3f}")
        print(f"  Order: {metrics['order_acc']:.3f}")
        print(f"  Estimated score: {score:.1f}")
        return
    
    if args.eval_valid:
        print("Evaluating on validation set...")
        all_preds = []
        all_gt = []
        
        for i, context in enumerate(read_jsonl(args.valid)):
            preds = predict_context(context)
            all_preds.extend(preds)
            for q in context['queries']:
                all_gt.append(q)
        
        metrics = evaluate(all_preds, all_gt)
        score = compute_score(metrics)
        print(f"Valid metrics:")
        print(f"  Total: {metrics['total']}")
        print(f"  Terminal: {metrics['terminal_acc']:.3f}")
        print(f"  Events: {metrics['event_acc']:.3f}")
        print(f"  Grid: {metrics['grid_acc']:.3f}")
        print(f"  Timeline: {metrics['timeline_acc']:.3f}")
        print(f"  Order: {metrics['order_acc']:.3f}")
        print(f"  Estimated score: {score:.1f}")
        return
    
    print("Generating test predictions...")
    all_preds = []
    
    for i, context in enumerate(read_jsonl(args.test)):
        preds = predict_context(context)
        all_preds.extend(preds)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} contexts...")
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)
    
    print(f"Wrote {len(all_preds)} predictions to {args.out}")


if __name__ == '__main__':
    main()
