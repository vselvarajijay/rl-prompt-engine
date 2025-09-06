#!/usr/bin/env python3
"""
Business-friendly conversation simulator that shows what the AI would actually say to customers.
This makes the RL system much easier to understand and use.
"""

import argparse
import random
from purchase_env import AutomotiveAppointmentEnv
from stable_baselines3 import PPO

# Real conversation templates for each action
CONVERSATION_TEMPLATES = {
    0: {  # Rapport
        "templates": [
            "Hi! I'm calling about the car you're interested in. How are you doing today?",
            "Hello! I hope you're having a great day. I'm following up on your inquiry about our vehicles.",
            "Good morning! I wanted to reach out personally about the car you looked at online."
        ]
    },
    1: {  # Qualify
        "templates": [
            "What kind of vehicle are you looking for?",
            "What's most important to you in a car - reliability, fuel economy, or performance?",
            "Are you looking for new or used? What's your budget range?",
            "What features are most important to you and your family?"
        ]
    },
    2: {  # Show Inventory
        "templates": [
            "We have some great options that might be perfect for you. Let me tell you about a few.",
            "I found a couple of vehicles that match what you're looking for. Would you like to hear about them?",
            "We have a {car_model} that's very popular with families like yours. It has {features}.",
            "I can show you some cars that fit your needs and budget. When would be a good time?"
        ]
    },
    3: {  # Handle Concerns
        "templates": [
            "I understand your concern about the price. Let me show you some financing options.",
            "That's a valid worry about reliability. This brand has an excellent warranty and reputation.",
            "I hear you on the cost. We have some incentives that might help with that.",
            "I understand your hesitation. Many customers have had the same concern, but they've been very happy."
        ]
    },
    4: {  # Create Urgency
        "templates": [
            "This model is very popular and we only have a few left. I'd hate for you to miss out.",
            "We have a limited-time financing offer that ends this week. It could save you quite a bit.",
            "This is the last one in this color, and it's been getting a lot of interest.",
            "Our current promotion is ending soon, and I'd hate for you to miss these savings."
        ]
    },
    5: {  # Social Proof
        "templates": [
            "We've sold 50 of these this month, and customers are loving them.",
            "This is our best-selling vehicle. Families like yours have been very happy with it.",
            "Many customers in your situation have chosen this car and they're thrilled with their decision.",
            "This model has excellent reviews and our customers keep coming back for more."
        ]
    },
    6: {  # Offer Incentives
        "templates": [
            "I can get you a great deal on this vehicle. Let me see what discounts I can apply.",
            "We're offering special financing rates right now that could save you money.",
            "I have some incentives that might interest you. Would you like to hear about them?",
            "I can offer you our current promotion plus some additional savings. It's a great opportunity."
        ]
    },
    7: {  # Book Appointment
        "templates": [
            "I can show you the car today - are you free this afternoon?",
            "Would you like to schedule a test drive? I have some time slots available.",
            "I can book you an appointment to see the vehicle. When works best for you?",
            "Let's get you in to see this car. What time today or tomorrow works for you?"
        ]
    }
}

# Car models and features for realistic examples
CAR_MODELS = ["Honda Civic", "Toyota Camry", "Ford F-150", "Chevrolet Equinox", "Nissan Altima"]
CAR_FEATURES = ["excellent fuel economy", "advanced safety features", "spacious interior", "reliable engine", "modern technology"]

def get_conversation_response(action: int, customer_state: dict) -> str:
    """Get a realistic conversation response for the given action."""
    if action not in CONVERSATION_TEMPLATES:
        return "I'd like to help you with your car needs."
    
    template = random.choice(CONVERSATION_TEMPLATES[action]["templates"])
    
    # Add some personalization based on customer state
    if "{car_model}" in template:
        template = template.replace("{car_model}", random.choice(CAR_MODELS))
    if "{features}" in template:
        template = template.replace("{features}", random.choice(CAR_FEATURES))
    
    return template

def simulate_conversation(model, customer_type: str = "random", max_turns: int = 10):
    """Simulate a realistic conversation with a customer."""
    print(f"🚗 Starting conversation with {customer_type} customer...")
    print("=" * 60)
    
    # Create environment
    env = AutomotiveAppointmentEnv(n_personas=6)
    obs, _ = env.reset()
    
    # Get customer type name
    customer_types = ["Cautious Buyer", "Price Shopper", "Ready Buyer", "Research Buyer", "Impulse Buyer", "Skeptical Buyer"]
    customer_name = customer_types[obs["persona_id"]]
    
    print(f"👤 Customer: {customer_name}")
    print(f"📊 Initial State: Interest={obs['features'][0]:.2f}, Trust={obs['features'][3]:.2f}, Commitment={obs['features'][4]:.2f}")
    print()
    
    turn = 0
    while turn < max_turns:
        # Get AI's action
        action, _ = model.predict(obs, deterministic=True)
        
        # Get conversation response
        response = get_conversation_response(int(action), obs)
        
        # Show the conversation
        print(f"🤖 AI: {response}")
        
        # Take action in environment
        obs, reward, done, truncated, info = env.step(int(action))
        
        # Show customer's psychological state
        features = obs['features']
        print(f"📊 Customer State: Interest={features[0]:.2f}, Urgency={features[1]:.2f}, Availability={features[2]:.2f}, Trust={features[3]:.2f}, Commitment={features[4]:.2f}")
        
        if done:
            if reward == 1.0:
                print("✅ SUCCESS: Customer booked an appointment!")
            else:
                print("❌ Customer didn't book an appointment.")
            break
        
        print()
        turn += 1
    
    print("=" * 60)
    return reward == 1.0

def run_conversation_demo(model, num_conversations: int = 5):
    """Run multiple conversation demonstrations."""
    print(f"🎭 Running {num_conversations} conversation demonstrations...")
    print()
    
    successes = 0
    for i in range(num_conversations):
        print(f"--- Conversation {i+1} ---")
        success = simulate_conversation(model)
        if success:
            successes += 1
        print()
    
    print(f"📊 Results: {successes}/{num_conversations} conversations successful ({successes/num_conversations*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Simulate realistic conversations with customers")
    parser.add_argument("--model", type=str, default="ppo_purchase_sparse", help="Model file to load")
    parser.add_argument("--conversations", type=int, default=3, help="Number of conversations to simulate")
    parser.add_argument("--turns", type=int, default=8, help="Maximum turns per conversation")
    
    args = parser.parse_args()
    
    print("🚗 Automotive Appointment Booking - Conversation Simulator")
    print("=" * 60)
    
    # Load the trained model
    try:
        model = PPO.load(args.model)
        print(f"✅ Loaded trained model: {args.model}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained a model first with: python train.py")
        return
    
    # Run conversation demonstrations
    run_conversation_demo(model, args.conversations)
    
    print("🎉 Conversation simulation complete!")
    print("\n💡 This shows what your AI would actually say to customers!")
    print("🔧 You can customize the conversation templates in this file.")

if __name__ == "__main__":
    main()
