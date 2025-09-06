#!/usr/bin/env python3
"""
Simple business interface for the automotive appointment booking AI.
This makes it easy for business users to test and understand the system.
"""

import json
from conversation_simulator import simulate_conversation, run_conversation_demo
from stable_baselines3 import PPO

def show_menu():
    """Display the main menu."""
    print("\n" + "="*60)
    print("🚗 AUTOMOTIVE APPOINTMENT BOOKING AI")
    print("="*60)
    print("1. 🎭 Watch AI have conversations with customers")
    print("2. 📊 Test AI performance (success rate)")
    print("3. ⚙️  View current configuration")
    print("4. 🔧 Quick configuration changes")
    print("5. ❓ Help - How does this work?")
    print("6. 🚪 Exit")
    print("="*60)

def watch_conversations():
    """Let user watch AI conversations."""
    print("\n🎭 CONVERSATION SIMULATOR")
    print("-" * 40)
    
    try:
        model = PPO.load("ppo_purchase_sparse")
        print("✅ AI model loaded successfully!")
        
        num_conv = input("How many conversations would you like to watch? (1-10): ")
        try:
            num_conv = int(num_conv)
            if num_conv < 1 or num_conv > 10:
                num_conv = 3
        except:
            num_conv = 3
        
        run_conversation_demo(model, num_conv)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've trained the AI first with: python train.py")

def test_performance():
    """Test AI performance."""
    print("\n📊 PERFORMANCE TEST")
    print("-" * 40)
    
    try:
        from eval import evaluate
        model = PPO.load("ppo_purchase_sparse")
        
        episodes = input("How many test conversations? (100-1000): ")
        try:
            episodes = int(episodes)
            if episodes < 100 or episodes > 1000:
                episodes = 500
        except:
            episodes = 500
        
        print(f"Testing AI with {episodes} conversations...")
        win_rate, action_counts = evaluate(model, episodes)
        
        print(f"\n📈 RESULTS:")
        print(f"Success Rate: {win_rate:.1%}")
        print(f"Total Conversations: {episodes}")
        
        print(f"\n🎯 AI Strategy (most used tactics):")
        total_actions = sum(action_counts.values())
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_actions) * 100
            action_names = {
                0: "Build Rapport", 1: "Ask Questions", 2: "Show Cars", 
                3: "Address Concerns", 4: "Create Urgency", 5: "Social Proof",
                6: "Offer Deals", 7: "Book Appointment"
            }
            print(f"  {action_names.get(action, f'Action {action}')}: {percentage:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've trained the AI first with: python train.py")

def view_configuration():
    """Show current configuration."""
    print("\n⚙️  CURRENT CONFIGURATION")
    print("-" * 40)
    
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        print("🎭 Customer Types:")
        for type_key, type_data in config["customer_types"].items():
            print(f"  {type_data['name']}: {type_data['description']}")
        
        print(f"\n📊 Environment Settings:")
        env = config["environment"]
        print(f"  Max conversation length: {env['max_conversation_turns']} turns")
        print(f"  Number of customer types: {env['num_customer_types']}")
        print(f"  Success threshold: {env['success_threshold']}")
        
        print(f"\n🎯 Sales Actions:")
        for action_name, action_data in config["sales_actions"].items():
            print(f"  {action_name}: {action_data['description']}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")

def quick_config_changes():
    """Allow quick configuration changes."""
    print("\n🔧 QUICK CONFIGURATION CHANGES")
    print("-" * 40)
    print("⚠️  Note: This is a simplified interface.")
    print("For advanced changes, edit config.json directly.")
    
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        print("\nCurrent success thresholds:")
        psych = config["customer_psychology"]
        print(f"  Interest: {psych['interest']['ready_threshold']}")
        print(f"  Trust: {psych['trust']['ready_threshold']}")
        print(f"  Availability: {psych['availability']['ready_threshold']}")
        print(f"  Commitment: {psych['commitment']['ready_threshold']}")
        
        print("\nTo make customers easier to convince (higher success rate):")
        print("  - Lower these thresholds (e.g., 0.3 instead of 0.5)")
        print("  - Edit config.json and change 'ready_threshold' values")
        
        print("\nTo make customers harder to convince (lower success rate):")
        print("  - Raise these thresholds (e.g., 0.7 instead of 0.5)")
        print("  - Edit config.json and change 'ready_threshold' values")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")

def show_help():
    """Show help information."""
    print("\n❓ HOW DOES THIS WORK?")
    print("-" * 40)
    print("""
🎯 WHAT THE AI DOES:
The AI learns to have conversations with potential car buyers and convince them to book appointments at your dealership.

🧠 HOW IT LEARNS:
1. The AI tries different sales tactics (rapport, showing cars, offering deals, etc.)
2. It learns which tactics work best for different types of customers
3. Over time, it gets better at booking appointments

👥 CUSTOMER TYPES:
- Cautious Buyer: Takes time to decide, needs lots of information
- Price Shopper: Very focused on getting the best deal  
- Ready Buyer: Already knows what they want, ready to buy
- Research Buyer: Wants to learn everything before deciding
- Impulse Buyer: Makes quick decisions, easy to convince
- Skeptical Buyer: Hard to convince, needs lots of proof

🎭 SALES TACTICS:
- Build Rapport: Build relationship and trust
- Ask Questions: Learn about customer needs
- Show Cars: Show available vehicles
- Address Concerns: Handle price/reliability worries
- Create Urgency: Create time pressure with limited offers
- Social Proof: Show others bought and are happy
- Offer Deals: Offer special deals and financing
- Book Appointment: Try to book the appointment

📊 UNDERSTANDING RESULTS:
- Success Rate: Percentage of customers who book appointments
- Action Distribution: Which tactics the AI uses most often
- Higher percentages = AI thinks that tactic works well

🔧 CUSTOMIZATION:
Edit config.json to:
- Change customer behavior
- Adjust sales tactic effectiveness  
- Add new customer types
- Modify success criteria
""")

def main():
    """Main business interface."""
    print("Welcome to the Automotive Appointment Booking AI!")
    
    while True:
        show_menu()
        choice = input("\nWhat would you like to do? (1-6): ").strip()
        
        if choice == "1":
            watch_conversations()
        elif choice == "2":
            test_performance()
        elif choice == "3":
            view_configuration()
        elif choice == "4":
            quick_config_changes()
        elif choice == "5":
            show_help()
        elif choice == "6":
            print("\n👋 Thanks for using the Automotive Appointment Booking AI!")
            break
        else:
            print("\n❌ Please enter a number between 1-6")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
