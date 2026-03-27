"""
Pursuit-Evasion Environment with Fog-of-War Partial Observability
Implements the POSSG formulation from B-SRM-CHFA paper
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import IntEnum

class Action(IntEnum):
    """5-action space: cardinal directions + stay"""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    STAY = 4

@dataclass
class AgentState:
    """Individual agent state"""
    position: Tuple[int, int]
    agent_id: int
    is_pursuer: bool
    
@dataclass
class Observation:
    """Local 5x5 observation with fog-of-war"""
    local_grid: np.ndarray  # 5x5x3 (pursuers, evaders, walls)
    self_position: Tuple[int, int]  # Agent's own position (relative to view)
    detected_opponents: List[Tuple[int, int]]  # Opponent positions in view
    timestep: int

class PursuitEvasionEnv:
    """
    Multi-agent pursuit-evasion with symmetric partial observability.
    
    State: positions of all agents (pursuers + evaders)
    Observation: 5x5 local neighborhood centered on agent
    Actions: {N, S, E, W, Stay}
    Termination: capture (pursuer adjacent to evader) OR timeout H_max
    Cost: pursuers pay +1 per timestep, evaders pay -1
    """
    
    def __init__(
        self,
        grid_size: int = 10,
        n_pursuers: int = 3,
        n_evaders: int = 2,
        observation_radius: int = 2,  # 5x5 = radius 2
        max_steps: int = 100,
        capture_radius: int = 1,  # Adjacent = captured
        terminal_penalty: float = 50.0,
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_agents = n_pursuers + n_evaders
        self.obs_radius = observation_radius
        self.max_steps = max_steps
        self.capture_radius = capture_radius
        self.terminal_penalty = terminal_penalty
        
        self.rng = np.random.default_rng(seed)
        
        # State tracking
        self.pursuers: List[AgentState] = []
        self.evaders: List[AgentState] = []
        self.timestep = 0
        self.done = False
        self.captured = False
        
        # Walls (obstacles) - can add later for complexity
        self.walls = np.zeros((grid_size, grid_size), dtype=bool)
        
    def reset(self) -> Tuple[List[Observation], List[Observation]]:
        """
        Reset environment, return initial observations for pursuers and evaders.
        
        Returns:
            (pursuer_observations, evader_observations)
        """
        self.timestep = 0
        self.done = False
        self.captured = False
        
        # Random initial positions (ensure no overlap)
        positions = self.rng.choice(
            self.grid_size * self.grid_size,
            size=self.n_agents,
            replace=False
        )
        
        # Initialize pursuers
        self.pursuers = [
            AgentState(
                position=self._idx_to_pos(positions[i]),
                agent_id=i,
                is_pursuer=True
            )
            for i in range(self.n_pursuers)
        ]
        
        # Initialize evaders
        self.evaders = [
            AgentState(
                position=self._idx_to_pos(positions[self.n_pursuers + i]),
                agent_id=i,
                is_pursuer=False
            )
            for i in range(self.n_evaders)
        ]
        
        # Get initial observations
        pursuer_obs = [self._get_observation(p) for p in self.pursuers]
        evader_obs = [self._get_observation(e) for e in self.evaders]
        
        return pursuer_obs, evader_obs
    
    def step(
        self,
        pursuer_actions: List[Action],
        evader_actions: List[Action]
    ) -> Tuple[List[Observation], List[Observation], List[float], List[float], bool]:
        """
        Execute one timestep.
        
        Args:
            pursuer_actions: List of actions for each pursuer
            evader_actions: List of actions for each evader
            
        Returns:
            pursuer_obs, evader_obs, pursuer_costs, evader_costs, done
        """
        assert len(pursuer_actions) == self.n_pursuers
        assert len(evader_actions) == self.n_evaders
        assert not self.done, "Environment is done, call reset()"
        
        # Execute actions (pursuers first, then evaders - or simultaneous)
        # Here we do simultaneous movement
        self._move_agents(self.pursuers, pursuer_actions)
        self._move_agents(self.evaders, evader_actions)
        
        self.timestep += 1
        
        # Check termination conditions
        captured = self._check_capture()
        timeout = self.timestep >= self.max_steps
        self.captured = captured
        self.done = captured or timeout
        
        # Compute costs
        if captured:
            # Pursuers succeeded early - pay only timesteps used
            pursuer_costs = [float(self.timestep)] * self.n_pursuers
            evader_costs = [-float(self.timestep)] * self.n_evaders
        elif timeout:
            # Pursuers failed - pay penalty
            pursuer_costs = [float(self.max_steps) + self.terminal_penalty] * self.n_pursuers
            evader_costs = [-(float(self.max_steps) + self.terminal_penalty)] * self.n_evaders
        else:
            # Normal timestep cost
            pursuer_costs = [1.0] * self.n_pursuers
            evader_costs = [-1.0] * self.n_evaders
        
        # Get observations
        pursuer_obs = [self._get_observation(p) for p in self.pursuers]
        evader_obs = [self._get_observation(e) for e in self.evaders]
        
        return pursuer_obs, evader_obs, pursuer_costs, evader_costs, self.done
    
    def _move_agents(self, agents: List[AgentState], actions: List[Action]):
        """Execute movement actions for a team"""
        for agent, action in zip(agents, actions):
            x, y = agent.position
            
            if action == Action.NORTH:
                new_pos = (x, max(0, y - 1))
            elif action == Action.SOUTH:
                new_pos = (x, min(self.grid_size - 1, y + 1))
            elif action == Action.EAST:
                new_pos = (min(self.grid_size - 1, x + 1), y)
            elif action == Action.WEST:
                new_pos = (max(0, x - 1), y)
            else:  # STAY
                new_pos = (x, y)
            
            # Check if new position is valid (not a wall, not occupied by same team)
            if not self.walls[new_pos] and not self._is_occupied_same_team(new_pos, agent):
                agent.position = new_pos
    
    def _is_occupied_same_team(self, pos: Tuple[int, int], moving_agent: AgentState) -> bool:
        """Check if position is occupied by another agent ON THE SAME TEAM.
        
        Cross-team occupation is ALLOWED — this enables capture by moving
        onto an opponent's cell. Only same-team collisions are blocked.
        """
        team = self.pursuers if moving_agent.is_pursuer else self.evaders
        for agent in team:
            if agent.agent_id != moving_agent.agent_id and agent.position == pos:
                return True
        return False
    
    def _check_capture(self) -> bool:
        """Check if any evader is within capture radius of any pursuer"""
        for pursuer in self.pursuers:
            for evader in self.evaders:
                dist = self._manhattan_distance(pursuer.position, evader.position)
                if dist <= self.capture_radius:
                    return True
        return False
    
    def _get_observation(self, agent: AgentState) -> Observation:
        """
        Generate local 5x5 observation centered on agent.
        Observation has 3 channels: [pursuers, evaders, walls]
        """
        x, y = agent.position
        radius = self.obs_radius
        
        # Initialize 5x5x3 observation
        local_grid = np.zeros((2*radius+1, 2*radius+1, 3), dtype=np.float32)
        
        # Populate grid
        detected_opponents = []
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                world_x, world_y = x + dx, y + dy
                
                # Out of bounds or wall
                if (world_x < 0 or world_x >= self.grid_size or 
                    world_y < 0 or world_y >= self.grid_size):
                    local_grid[dy+radius, dx+radius, 2] = 1.0  # Wall channel
                    continue
                
                if self.walls[world_x, world_y]:
                    local_grid[dy+radius, dx+radius, 2] = 1.0
                    continue
                
                # Check for agents
                for pursuer in self.pursuers:
                    if pursuer.position == (world_x, world_y):
                        local_grid[dy+radius, dx+radius, 0] = 1.0
                        if not agent.is_pursuer:
                            detected_opponents.append((dx, dy))
                
                for evader in self.evaders:
                    if evader.position == (world_x, world_y):
                        local_grid[dy+radius, dx+radius, 1] = 1.0
                        if agent.is_pursuer:
                            detected_opponents.append((dx, dy))
        
        return Observation(
            local_grid=local_grid,
            self_position=(radius, radius),  # Center of view
            detected_opponents=detected_opponents,
            timestep=self.timestep
        )
    
    def get_state(self) -> np.ndarray:
        """
        Get global state (for centralized training only).
        
        Returns:
            state vector: [pursuer_positions..., evader_positions..., timestep]
        """
        state = np.zeros(2 * self.n_agents + 1, dtype=np.float32)
        
        for i, pursuer in enumerate(self.pursuers):
            state[2*i] = pursuer.position[0] / self.grid_size
            state[2*i + 1] = pursuer.position[1] / self.grid_size
        
        offset = 2 * self.n_pursuers
        for i, evader in enumerate(self.evaders):
            state[offset + 2*i] = evader.position[0] / self.grid_size
            state[offset + 2*i + 1] = evader.position[1] / self.grid_size
        
        state[-1] = self.timestep / self.max_steps
        
        return state
    
    def _idx_to_pos(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (x, y) position"""
        return (idx % self.grid_size, idx // self.grid_size)
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self, mode='human'):
        """Simple ASCII rendering for debugging"""
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        
        # Add walls
        grid[self.walls] = '#'
        
        # Add agents
        for pursuer in self.pursuers:
            x, y = pursuer.position
            grid[y, x] = 'P'
        
        for evader in self.evaders:
            x, y = evader.position
            grid[y, x] = 'E'
        
        print(f"\n=== Step {self.timestep}/{self.max_steps} ===")
        for row in grid:
            print(' '.join(row))
        print()


if __name__ == "__main__":
    # Test environment
    env = PursuitEvasionEnv(grid_size=10, n_pursuers=3, n_evaders=2, max_steps=50)
    
    pursuer_obs, evader_obs = env.reset()
    print(f"Initialized with {env.n_pursuers} pursuers, {env.n_evaders} evaders")
    env.render()
    
    # Run random episode
    for step in range(10):
        # Random actions
        pursuer_actions = [Action(env.rng.integers(0, 5)) for _ in range(env.n_pursuers)]
        evader_actions = [Action(env.rng.integers(0, 5)) for _ in range(env.n_evaders)]
        
        p_obs, e_obs, p_costs, e_costs, done = env.step(pursuer_actions, evader_actions)
        
        env.render()
        print(f"Pursuer costs: {p_costs}, Evader costs: {e_costs}")
        
        if done:
            print("Episode terminated!")
            break
