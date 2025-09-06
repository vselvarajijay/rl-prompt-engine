import gymnasium as gym
import numpy as np
from gymnasium import spaces
from config import AutomotiveConfig

# Constants that define the environment structure
N_ACTIONS = 8  # number of different appointment booking tactics
MAX_TURNS = 12  # maximum conversation length - short episodes help sparse reward


class AutomotiveAppointmentEnv(gym.Env):
    """
    A Reinforcement Learning Environment for Automotive Appointment Booking

    This environment simulates a voice AI conversation where an AI agent tries to
    convince a potential car buyer to book an appointment with a salesperson.
    The agent can take different actions (appointment booking tactics) and receives
    feedback based on whether the customer books an appointment.

    RL Concepts Explained:
    - Environment: The "world" where the agent operates (this simulator)
    - Agent: The AI that learns to make decisions (uses this environment)
    - State/Observation: What the agent can see about the current situation
    - Action: What the agent can do at each step
    - Reward: Feedback that tells the agent how well it's doing
    - Episode: One complete conversation from start to finish

    Observation (Dict) - What the agent can observe about the current state:
      - turn_idx: [0..MAX_TURNS] scalar float32 - which turn of conversation
      - last_action: Discrete(N_ACTIONS+1) (+1 for 'none' at t=0)
      - persona_id: Discrete(n_personas) - customer personality type
      - features: 5 floats in [0,1] = [interest, urgency, availability, trust, commitment]

    Actions: Discrete(N_ACTIONS) - 8 different appointment booking tactics
    Reward: only at episode end: 1 if appointment booked, else 0 - sparse reward
    """
    metadata = {"render_modes": []}

    def __init__(self, n_personas: int = 6, seed: int | None = None, config_file: str = "config.json"):
        """
        Initialize the automotive appointment booking environment.
        
        Args:
            n_personas: Number of different customer personality types to simulate
            seed: Random seed for reproducible behavior (useful for testing)
            config_file: Path to configuration file for easy customization
        """
        super().__init__()
        self.n_personas = int(n_personas)
        self.rng = np.random.default_rng(seed)  # Random number generator for stochastic behavior
        
        # Load configuration
        self.config = AutomotiveConfig(config_file)
        self.max_turns = self.config.config["environment"]["max_conversation_turns"]

        # Define the observation space - what information the agent can observe
        # In RL, the observation space defines the "state" that the agent sees
        self.observation_space = spaces.Dict({
            "turn_idx": spaces.Box(low=0, high=self.max_turns, shape=(1,), dtype=np.float32),
            "last_action": spaces.Discrete(N_ACTIONS + 1),  # +1 for "no action" at start
            "persona_id": spaces.Discrete(self.n_personas),
            "features": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),  # 5 features for appointment booking
        })
        
        # Define the action space - what actions the agent can take
        # In RL, this defines all possible choices the agent can make
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Load action effects from configuration
        self.action_effects = np.array(self.config.get_action_effects(), dtype=np.float32)
        self.BOOK_APPOINTMENT = 7  # Special action for attempting to book appointment

        # Load customer personality types from configuration
        self.persona_priors = np.array(self.config.get_persona_priors(), dtype=np.float32)
        self.reset()  # Initialize the environment for the first episode


    def reset(self, seed: int | None = None, options=None):
        """
        Reset the environment to start a new appointment booking conversation episode.
        
        In RL, this is called at the beginning of each episode to start fresh.
        It randomly selects a customer persona and initializes their psychological state.
        
        Args:
            seed: Optional random seed for reproducible behavior
            options: Additional options (not used in this environment)
            
        Returns:
            observation: The initial state the agent sees
            info: Additional information (empty dict in this case)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Start a new conversation
        self.turn = 0  # Reset conversation turn counter
        self.persona_id = self.rng.integers(0, self.n_personas)  # Random customer type
        self.features = self.persona_priors[self.persona_id].copy()  # Init state
        self.last_action = N_ACTIONS  # "none" - no previous action at the start

        return self._obs(), {}  # Return initial observation and empty info dict

    def step(self, action: int):
        """
        Execute one step of the appointment booking conversation.
        
        This is the core RL function that:
        1. Takes an action from the agent
        2. Updates the customer's psychological state based on that action
        3. Determines if the customer books an appointment (and episode ends)
        4. Returns the new state, reward, and whether the episode is done
        
        Args:
            action: Integer from 0 to N_ACTIONS-1 representing the appointment booking tactic to use
            
        Returns:
            observation: The new state after taking the action
            reward: The reward for this step (usually 0, except at episode end)
            terminated: Whether the episode ended (appointment booked or max turns reached)
            truncated: Whether the episode was cut off early (not used in this environment)
            info: Additional information about the current state
        """
        self.turn += 1  # Move to the next turn of the conversation

        # Apply the action's effect to the customer's psychological state
        # Add some randomness to make the environment more realistic
        noise = self.rng.normal(0.0, 0.01, size=5).astype(np.float32)  # 5 features now
        self.features = np.clip(
            self.features + self.action_effects[action] + noise,
            0.0, 1.0  # Keep all features between 0 and 1
        )

        # Check if the customer books an appointment during this turn
        appointment_booked_now = False
        if action == self.BOOK_APPOINTMENT and self._is_ready_to_book():
            # If we tried to book and the customer is ready, they might book
            # Give a small boost to booking probability when actively trying to book
            appointment_booked_now = self.rng.random() < self._appointment_prob(boost=0.05)

        # Determine if the episode is over
        terminated = bool(appointment_booked_now or self.turn >= self.max_turns)
        truncated = False  # This environment doesn't use truncation

        # Calculate reward (this is a sparse reward environment)
        if terminated:
            # Only give reward at the end of the episode
            p = self._appointment_prob()  # Calculate probability of booking appointment
            reward = float(self.rng.random() < p)  # 1 if books appointment, 0 if not
        else:
            reward = 0.0  # No reward during conversation, only at the end

        # Prepare information about the current state
        info = {
            "appointment_prob": float(self._appointment_prob()),
            "appointment_booked": bool(terminated and reward == 1.0),
        }

        self.last_action = action  # Remember this action for the next observation
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        """
        Create the observation that the agent sees.
        
        In RL, the observation is the "state" that the agent uses to make decisions.
        This function packages up all the relevant information about the current
        conversation state into a format the agent can understand.
        
        Returns:
            Dictionary containing:
            - turn_idx: Which turn of the conversation we're on
            - last_action: What action was taken in the previous turn
            - persona_id: Which customer personality type we're dealing with
            - features: Customer's current state [interest, urgency, availability, trust, commitment]
        """
        return {
            "turn_idx": np.array([self.turn], dtype=np.float32),
            "last_action": int(self.last_action),
            "persona_id": int(self.persona_id),
            "features": self.features.astype(np.float32),
        }

    def _is_ready_to_book(self):
        """
        Check if the customer is psychologically ready to book an appointment.
        
        This function implements the business logic for when a customer is likely
        to be receptive to booking an appointment based on configurable thresholds.
        
        Returns:
            Boolean indicating if the customer is ready to book an appointment
        """
        interest, urgency, availability, trust, commitment = self.features
        
        # Get thresholds from configuration
        thresholds = self.config.config["customer_psychology"]
        return ((interest > thresholds["interest"]["ready_threshold"]) and 
                (trust > thresholds["trust"]["ready_threshold"]) and
                (availability > thresholds["availability"]["ready_threshold"]) and 
                (commitment > thresholds["commitment"]["ready_threshold"]))

    def _appointment_prob(self, boost: float = 0.0):
        """
        Calculate the probability that the customer will book an appointment.
        
        This function uses a logistic regression model to determine appointment
        booking probability based on the customer's current psychological state. The
        formula combines:
        - Interest level (positive effect - more interest = higher probability)
        - Trust level (positive effect - more trust = higher probability)
        - Availability (positive effect - more flexible schedule = higher probability)
        - Commitment (positive effect - more ready to commit = higher probability)
        - Urgency (negative effect - too much urgency can be off-putting)

        The logistic function (1 / (1 + e^(-z))) converts the linear combination
        into a probability between 0 and 1.
        
        Args:
            boost: Additional probability boost (e.g., when actively trying to book)
            
        Returns:
            Probability of booking appointment between 0.0 and 1.0
        """
        interest, urgency, availability, trust, commitment = self.features
        
        # Linear combination of psychological features
        # Higher interest, trust, availability, and commitment increase probability
        # Higher urgency can decrease probability (too pushy)
        z = (1.5 * interest      # Interest has strong positive effect
             + 1.2 * trust       # Trust has strong positive effect
             + 1.0 * availability  # Availability has positive effect
             + 1.8 * commitment  # Commitment has strongest positive effect
             - 0.5 * urgency     # Urgency has negative effect (too pushy)
             + 0.1               # Base probability offset
             + boost)            # Additional boost (e.g., when booking)

        # Convert to probability using logistic function
        return 1.0 / (1.0 + np.exp(-z))
